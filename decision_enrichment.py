from __future__ import annotations

import json
from pathlib import Path

from business_rules import evaluate_business_rules

EVAL_RESULTS_PATH = Path("eval_results.json")


def enrich_record(record: dict) -> dict:
    agent_result = record.get("agent_result") or {}
    columns = agent_result.get("columns") or []
    rows = agent_result.get("rows") or []
    rows_as_tuples = [tuple(row) for row in rows]

    decision = evaluate_business_rules(
        question=record.get("question", ""),
        columns=columns,
        rows=rows_as_tuples,
        sql=record.get("agent_sql") or "",
    )
    record.update(decision)
    return record


def main() -> None:
    if not EVAL_RESULTS_PATH.exists():
        raise FileNotFoundError("eval_results.json not found. Run python eval_harness.py first.")

    with EVAL_RESULTS_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    enriched = [enrich_record(record) for record in records]

    with EVAL_RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2)

    rule_counts = {}
    for record in enriched:
        rule = record.get("recommendation_rule") or "unknown"
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

    print(f"Enriched {len(enriched)} eval records with business recommendation fields.")
    print("Recommendation rule breakdown:")
    for rule, count in sorted(rule_counts.items()):
        print(f"  {rule}: {count}")


if __name__ == "__main__":
    main()
