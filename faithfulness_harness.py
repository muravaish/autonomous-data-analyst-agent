import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

EVAL_RESULTS_PATH = Path("eval_results.json")
FAITHFULNESS_RESULTS_PATH = Path("faithfulness_results.json")

# Matches standalone numbers, including comma thousands and percentages, while
# avoiding digits embedded inside product/order ids such as d6160fb...
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![A-Za-z0-9_])")


def parse_number(raw: str) -> float | None:
    """Convert a text number like '50,326.18' or '12.5%' to a float."""
    if not raw:
        return None
    cleaned = raw.replace(",", "").strip()
    is_percent = cleaned.endswith("%")
    cleaned = cleaned.rstrip("%")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value


def extract_text_numbers(text: str | None) -> list[dict[str, Any]]:
    """Extract numeric mentions from generated insight text."""
    if not text:
        return []

    mentions = []
    for match in NUMBER_RE.finditer(text):
        raw = match.group(0)
        value = parse_number(raw)
        if value is None:
            continue
        mentions.append({"raw": raw, "value": value, "is_percent": raw.endswith("%"), "start": match.start(), "end": match.end()})
    return mentions


def flatten_numeric_values(value: Any) -> list[float]:
    """Collect numeric values from nested result data without treating ids as numbers."""
    values: list[float] = []

    if isinstance(value, bool) or value is None:
        return values
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
            return [float(stripped)]
        return values
    if isinstance(value, dict):
        for nested in value.values():
            values.extend(flatten_numeric_values(nested))
        return values
    if isinstance(value, list):
        for nested in value:
            values.extend(flatten_numeric_values(nested))
        return values

    return values


def result_numeric_values(record: dict[str, Any]) -> list[float]:
    """Use the actual rows returned by the agent SQL as the grounding source."""
    agent_result = record.get("agent_result") or {}
    rows = agent_result.get("rows") if isinstance(agent_result, dict) else None
    return flatten_numeric_values(rows or [])


def question_context_values(record: dict[str, Any]) -> set[float]:
    """Numbers from the question are context, not claims made from the SQL result."""
    return {mention["value"] for mention in extract_text_numbers(record.get("question", ""))}


def numbers_close(claimed: float, actual: float) -> bool:
    """Rounding-tolerant comparison for narrative numbers."""
    abs_claimed = abs(claimed)
    tolerance = max(0.05, abs_claimed * 0.01)
    return abs(claimed - actual) <= tolerance


def mention_matches_value(mention: dict[str, Any], actual: float) -> bool:
    """Percent mentions may correspond to either 32.09 or 0.3209 in data."""
    candidates = [mention["value"]]
    if mention.get("is_percent"):
        candidates.append(mention["value"] / 100.0)
    return any(numbers_close(candidate, actual) for candidate in candidates)


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    insight_mentions = extract_text_numbers(record.get("insight"))
    data_values = result_numeric_values(record)
    context_values = question_context_values(record)

    checked = []
    grounded_count = 0
    ignored_context_count = 0

    for mention in insight_mentions:
        value = mention["value"]
        if any(mention_matches_value(mention, context) for context in context_values):
            status = "ignored_context_number"
            ignored_context_count += 1
            matched_value = None
        else:
            matched = [actual for actual in data_values if mention_matches_value(mention, actual)]
            if matched:
                status = "grounded"
                grounded_count += 1
                matched_value = matched[0]
            else:
                status = "possible_hallucination"
                matched_value = None

        checked.append({
            "raw": mention["raw"],
            "value": value,
            "status": status,
            "matched_data_value": matched_value,
        })

    checked_count = len(insight_mentions) - ignored_context_count
    hallucinations = [item for item in checked if item["status"] == "possible_hallucination"]
    faithful = len(hallucinations) == 0
    score = grounded_count / checked_count if checked_count else 1.0

    return {
        "id": record.get("id"),
        "difficulty": record.get("difficulty"),
        "question": record.get("question"),
        "execution_match": record.get("match"),
        "faithful": faithful,
        "faithfulness_score": round(score, 3),
        "numbers_mentioned": len(insight_mentions),
        "numbers_checked": checked_count,
        "numbers_grounded": grounded_count,
        "numbers_ignored_as_context": ignored_context_count,
        "possible_hallucinations": hallucinations,
        "checked_numbers": checked,
        "insight": record.get("insight"),
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)
    faithful_count = sum(1 for item in results if item["faithful"])
    checked_total = sum(item["numbers_checked"] for item in results)
    grounded_total = sum(item["numbers_grounded"] for item in results)
    numeric_score = grounded_total / checked_total if checked_total else 1.0

    print("\n" + "=" * 70)
    print("FAITHFULNESS CHECK RESULTS")
    print("=" * 70)

    for item in results:
        status = "PASS" if item["faithful"] else "FLAGGED"
        print(
            f"{item['id']} [{item['difficulty']}] {status}: "
            f"{item['numbers_grounded']}/{item['numbers_checked']} checked numbers grounded "
            f"({item['numbers_mentioned']} mentioned, {item['numbers_ignored_as_context']} context ignored)"
        )
        for hallucination in item["possible_hallucinations"]:
            print(f"  possible hallucination: {hallucination['raw']} (parsed {hallucination['value']})")

    print("\n" + "=" * 70)
    print(f"OVERALL FAITHFUL CASES: {faithful_count}/{total} ({100 * faithful_count / total:.1f}%)")
    print(f"OVERALL NUMERIC GROUNDING: {grounded_total}/{checked_total} ({100 * numeric_score:.1f}%)")
    print("=" * 70)

    by_difficulty = defaultdict(lambda: {"faithful": 0, "total": 0, "grounded": 0, "checked": 0})
    for item in results:
        bucket = by_difficulty[item["difficulty"]]
        bucket["total"] += 1
        bucket["faithful"] += int(item["faithful"])
        bucket["grounded"] += item["numbers_grounded"]
        bucket["checked"] += item["numbers_checked"]

    print("\nBy difficulty:")
    for difficulty, stats in by_difficulty.items():
        case_pct = 100 * stats["faithful"] / stats["total"] if stats["total"] else 0.0
        numeric_pct = 100 * stats["grounded"] / stats["checked"] if stats["checked"] else 100.0
        print(
            f"  {difficulty:>8}: faithful cases {stats['faithful']}/{stats['total']} ({case_pct:.1f}%), "
            f"numeric grounding {stats['grounded']}/{stats['checked']} ({numeric_pct:.1f}%)"
        )

    flagged = [item for item in results if not item["faithful"]]
    if flagged:
        print("\nFlagged questions:")
        for item in flagged:
            nums = ", ".join(h["raw"] for h in item["possible_hallucinations"])
            print(f"  {item['id']}: {nums}")
    else:
        print("\nNo possible numeric hallucinations flagged.")


def main() -> None:
    if not EVAL_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {EVAL_RESULTS_PATH}. Run python eval_harness.py first."
        )

    with EVAL_RESULTS_PATH.open("r", encoding="utf-8") as f:
        eval_results = json.load(f)

    faithfulness_results = [evaluate_record(record) for record in eval_results]
    with FAITHFULNESS_RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(faithfulness_results, f, indent=2)

    print_summary(faithfulness_results)
    print(f"\nFull faithfulness results saved to {FAITHFULNESS_RESULTS_PATH}")


if __name__ == "__main__":
    main()


