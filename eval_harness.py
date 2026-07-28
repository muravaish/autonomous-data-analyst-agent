import json
import time
from agent_graph import build_graph

GOLD_PATH = "gold_qa.json"
RESULTS_PATH = "eval_results.json"


def load_gold_set():
    with open(GOLD_PATH, "r") as f:
        return json.load(f)



def load_existing_results():
    try:
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
def normalize_value(v):
    """Make a value comparable regardless of type quirks (float rounding,
    string case/whitespace, etc.)."""
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return v.strip().lower()
    return v


def extract_value_set(row_values):
    """Turn a row (or the gold expected_result dict's values) into a
    normalized, comparable set."""
    return set(normalize_value(v) for v in row_values)


def check_execution_match(agent_columns, agent_rows, gold_expected_result: dict):
    """
    Execution accuracy check: does ANY row returned by the agent contain
    the same set of values as the gold expected result?

    We check "any row" rather than "the first row" because a correct
    agent might return the right row in a different position (e.g. if
    it didn't LIMIT 1, or ordered differently) -- we care about whether
    the correct answer is present and correctly identified as the top
    result, so we specifically check row 0 first, then fall back to
    checking if it exists anywhere (which we treat as a partial signal).
    """
    if not agent_rows:
        return False, "no_rows_returned"

    gold_values = extract_value_set(gold_expected_result.values())

    def values_match(row_values):
        """Subset match in either direction: handles cases where gold has
        extra reference fields the question didn't strictly require (or
        vice versa), while still requiring real overlap on the core answer."""
        if not row_values or not gold_values:
            return False
        return row_values.issubset(gold_values) or gold_values.issubset(row_values)

    # Primary check: does the FIRST row match? (this is what actually matters
    # for "the agent's top answer is correct")
    first_row_values = extract_value_set(agent_rows[0])
    if values_match(first_row_values):
        return True, "exact_match_top_row"

    # Secondary check: is a matching row present anywhere else in the results?
    for row in agent_rows[1:]:
        if values_match(extract_value_set(row)):
            return False, "correct_value_present_but_not_top_row"

    return False, "no_matching_row_found"


def classify_failure(gold_item, agent_state, match_reason):
    """
    Basic failure taxonomy. This is intentionally simple for now --
    it inspects the generated SQL and error state to guess a category.
    Will get more sophisticated as more failure examples are collected.
    """
    if agent_state.get("error"):
        err = agent_state["error"].lower()
        if "unknown table" in err or "validation failed" in err:
            return "hallucinated_table_or_column"
        if "execution failed" in err:
            return "sql_execution_error"
        return "unknown_error"

    if match_reason == "no_rows_returned":
        return "empty_result_wrong_filter"
    if match_reason == "correct_value_present_but_not_top_row":
        return "wrong_ordering_or_missing_limit"
    if match_reason == "no_matching_row_found":
        # crude heuristics based on gold SQL vs agent SQL keyword differences
        gold_sql = gold_item["sql"].lower()
        agent_sql = agent_state.get("sql", "").lower()

        gold_joins = set(w for w in gold_sql.split() if w == "join")
        agent_joins = set(w for w in agent_sql.split() if w == "join")
        if len(gold_joins) != len(agent_joins):
            return "wrong_join"

        agg_words = ["sum(", "avg(", "count(", "max(", "min("]
        gold_aggs = [a for a in agg_words if a in gold_sql]
        agent_aggs = [a for a in agg_words if a in agent_sql]
        if gold_aggs != agent_aggs:
            return "wrong_aggregation"

        return "misread_question_or_other"

    return "uncategorized"


def run_evaluation():
    gold_set = load_gold_set()
    app = build_graph()

    results = load_existing_results()
    completed_ids = {row.get("id") for row in results}

    for item in gold_set:
        if item["id"] in completed_ids:
            print(f"Skipping {item['id']} -- already saved in {RESULTS_PATH}")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating {item['id']} [{item['difficulty']}]: {item['question']}")
        print("=" * 60)

        initial_state = {
            "question": item["question"],
            "plan": "",
            "sql": "",
            "validation_ok": False,
            "validation_message": "",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "insight": "",
            "detected_kpis": [],
            "difficulty_label": "",
            "sql_explanation": [],
            "recommendations": [],
            "recommendation": "",
            "priority": "",
            "recommendation_rule": "",
            "recommendation_reason": "",
            "recommended_action": "",
            "safety_status": {},
            "chart_path": None,
            "chart_type": None,
            "error": None,
        }

        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            print(f"  PIPELINE CRASHED: {e}")
            results.append({
                "id": item["id"],
                "difficulty": item["difficulty"],
                "question": item["question"],
                "match": False,
                "match_reason": "pipeline_crash",
                "failure_category": "pipeline_crash",
                "agent_sql": None,
                "gold_sql": item["sql"],
                "agent_result": None,
                "gold_result": item["expected_result"],
                "insight": None,
                "detected_kpis": [],
                "difficulty_label": "Unknown",
                "sql_explanation": [],
                "recommendations": [],
                "recommendation": None,
                "priority": None,
                "recommendation_rule": None,
                "recommendation_reason": None,
                "recommended_action": None,
                "safety_status": {},
            })
            continue

        match, reason = check_execution_match(
            final_state.get("columns", []),
            final_state.get("rows", []),
            item["expected_result"]
        )

        failure_category = None if match else classify_failure(item, final_state, reason)

        print(f"  Agent SQL: {final_state.get('sql')}")
        print(f"  Agent result: {final_state.get('rows')}")
        print(f"  Gold result:  {item['expected_result']}")
        print(f"  MATCH: {match}  (reason: {reason})")
        print(f"  Recommendation: {final_state.get('recommendation')} ({final_state.get('priority')})")
        print(f"  Rule: {final_state.get('recommendation_rule')}")
        if failure_category:
            print(f"  Failure category: {failure_category}")

        results.append({
            "id": item["id"],
            "difficulty": item["difficulty"],
            "question": item["question"],
            "match": match,
            "match_reason": reason,
            "failure_category": failure_category,
            "agent_sql": final_state.get("sql"),
            "gold_sql": item["sql"],
            "agent_result": {
                "columns": final_state.get("columns"),
                "rows": final_state.get("rows"),
            },
            "gold_result": item["expected_result"],
            "insight": final_state.get("insight"),
            "detected_kpis": final_state.get("detected_kpis", []),
            "difficulty_label": final_state.get("difficulty_label"),
            "sql_explanation": final_state.get("sql_explanation", []),
            "recommendations": final_state.get("recommendations", []),
            "recommendation": final_state.get("recommendation"),
            "priority": final_state.get("priority"),
            "recommendation_rule": final_state.get("recommendation_rule"),
            "recommendation_reason": final_state.get("recommendation_reason"),
            "recommended_action": final_state.get("recommended_action"),
            "safety_status": final_state.get("safety_status", {}),
        })

        save_results(results)

        # Small pause to be gentle on the free-tier API between questions
        time.sleep(2)

    # --- Summary ---
    total = len(results)
    correct = sum(1 for r in results if r["match"])
    print(f"\n\n{'='*60}")
    print(f"OVERALL EXECUTION ACCURACY: {correct}/{total} ({100*correct/total:.1f}%)")
    print("=" * 60)

    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        by_difficulty.setdefault(d, {"correct": 0, "total": 0})
        by_difficulty[d]["total"] += 1
        if r["match"]:
            by_difficulty[d]["correct"] += 1

    for d, stats in by_difficulty.items():
        pct = 100 * stats["correct"] / stats["total"]
        print(f"  {d:>8}: {stats['correct']}/{stats['total']} ({pct:.1f}%)")

    failure_counts = {}
    for r in results:
        if not r["match"]:
            cat = r["failure_category"]
            failure_counts[cat] = failure_counts.get(cat, 0) + 1

    if failure_counts:
        print("\nFailure breakdown:")
        for cat, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()







