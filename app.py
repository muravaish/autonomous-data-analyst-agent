import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agent_graph import build_graph

BASE_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_PATH = BASE_DIR / "eval_results.json"
FAITHFULNESS_RESULTS_PATH = BASE_DIR / "faithfulness_results.json"

SAMPLE_QUESTIONS = [
    "What is the average review score for each payment type?",
    "Which sellers have the highest late delivery risk?",
    "Which sellers generated the highest total revenue?",
    "What is the average freight cost as a percentage of item price?",
    "How many orders were delivered later than their estimated delivery date?",
]

FOLLOW_UP_MARKERS = ["them", "those", "that", "these", "top 5", "only", "from them"]


def initial_state(question: str) -> dict:
    return {
        "question": question,
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


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


def resolve_follow_up(question: str, history: list[dict]) -> str:
    if not history:
        return question
    lowered = question.lower()
    if not any(marker in lowered for marker in FOLLOW_UP_MARKERS):
        return question
    previous = history[0]
    previous_question = previous.get("question", "")
    previous_columns = previous.get("columns", [])
    previous_rows = previous.get("rows", [])[:5]
    return (
        f"Previous question: {previous_question}\n"
        f"Previous result columns: {previous_columns}\n"
        f"Previous top rows: {previous_rows}\n"
        f"Follow-up request: {question}\n"
        "Answer the follow-up using the previous result context where words like them, those, or these refer to the previous result."
    )


def run_agent(question: str) -> dict:
    app = get_graph()
    return app.invoke(initial_state(question))


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def result_dataframe(result: dict) -> pd.DataFrame:
    columns = result.get("columns") or []
    rows = result.get("rows") or []
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def clean_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def percent(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def eval_summary(eval_results: list[dict]) -> dict:
    total = len(eval_results)
    passed = sum(1 for row in eval_results if row.get("match"))
    by_difficulty = {}
    failures = {}
    for row in eval_results:
        difficulty = row.get("difficulty", "unknown")
        bucket = by_difficulty.setdefault(difficulty, {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(bool(row.get("match")))
        if not row.get("match"):
            category = row.get("failure_category") or "unclassified"
            failures[category] = failures.get(category, 0) + 1
    return {"total": total, "passed": passed, "accuracy": percent(passed, total), "by_difficulty": by_difficulty, "failures": failures}


def faithfulness_summary(faithfulness_results: list[dict]) -> dict:
    total = len(faithfulness_results)
    faithful = sum(1 for row in faithfulness_results if row.get("faithful"))
    checked = sum(row.get("numbers_checked", 0) for row in faithfulness_results)
    grounded = sum(row.get("numbers_grounded", 0) for row in faithfulness_results)
    return {"total": total, "faithful": faithful, "case_score": percent(faithful, total), "checked": checked, "grounded": grounded, "numeric_score": percent(grounded, checked)}


def difficulty_dataframe(summary: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "difficulty": clean_label(difficulty),
            "accuracy": percent(stats["correct"], stats["total"]),
            "correct": stats["correct"],
            "total": stats["total"],
        }
        for difficulty, stats in summary["by_difficulty"].items()
    )


def failure_dataframe(failures: dict) -> pd.DataFrame:
    if not failures:
        return pd.DataFrame(columns=["failure_category", "count"])
    return pd.DataFrame({"failure_category": clean_label(category), "count": count} for category, count in failures.items())


def recommendation_rule_dataframe(eval_results: list[dict]) -> pd.DataFrame:
    counts = {}
    for row in eval_results:
        rule = row.get("recommendation_rule") or "not_available"
        counts[rule] = counts.get(rule, 0) + 1
    return pd.DataFrame({"recommendation_rule": clean_label(rule), "count": count} for rule, count in counts.items())


def merged_eval_table(eval_results: list[dict], faithfulness_results: list[dict] | None) -> pd.DataFrame:
    faithfulness_by_id = {row.get("id"): row for row in faithfulness_results or []}
    rows = []
    for row in eval_results:
        faithful_row = faithfulness_by_id.get(row.get("id"), {})
        rows.append({
            "id": row.get("id"),
            "difficulty": row.get("difficulty"),
            "execution_pass": bool(row.get("match")),
            "faithful": faithful_row.get("faithful"),
            "failure_category": row.get("failure_category") or "",
            "match_reason": row.get("match_reason"),
            "question": row.get("question"),
        })
    return pd.DataFrame(rows)


def render_chart_from_state(state: dict):
    chart_path = state.get("chart_path")
    if not chart_path:
        st.info("No chart was generated for this result shape.")
        return
    path = Path(chart_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if path.exists():
        components.html(path.read_text(encoding="utf-8"), height=520, scrolling=True)
    else:
        st.warning(f"Chart file not found: {path}")


def build_report(state: dict) -> str:
    df = result_dataframe(state)
    explanation = "\n".join(f"- {item}" for item in state.get("sql_explanation", [])) or "- Not available"
    safety = state.get("safety_status") or {}
    safety_lines = "\n".join(f"- {clean_label(k)}: {v}" for k, v in safety.items()) or "- Not available"
    table_md = "```text" + "\n" + df.to_string(index=False) + "\n" + "```" if not df.empty else "No rows returned."
    return f"""# Business Analysis Report

## Question
{state.get('question', '')}

## Generated SQL
```sql
{state.get('sql', '')}
```

## Result
{table_md}

## Insight
{state.get('insight', '')}

## Recommendation
- Recommendation: {state.get('recommendation', '')}
- Priority: {state.get('priority', '')}
- Rule: {state.get('recommendation_rule', '')}
- Reason: {state.get('recommendation_reason', '')}
- Action: {state.get('recommended_action', '')}

## Why This SQL Was Used
{explanation}

## Safety Status
{safety_lines}
"""


def render_recommendation(state: dict):
    priority = state.get("priority") or "None"
    if priority == "High":
        st.error(f"{state.get('recommendation')} - High Priority")
    elif priority == "Medium":
        st.warning(f"{state.get('recommendation')} - Medium Priority")
    else:
        st.success(f"{state.get('recommendation') or 'Monitor KPI'} - {priority} Priority")
    st.write(state.get("recommendation_reason") or "No rule reason returned.")
    st.markdown("**Recommended action**")
    st.write(state.get("recommended_action") or "No action returned.")


st.set_page_config(page_title="Autonomous Data Analyst Agent", layout="wide")
st.title("Autonomous Data Analyst Agent")
st.caption("Agentic e-commerce analytics with validated SQL, grounded insights, and deterministic business recommendations")

chat_tab, eval_tab = st.tabs(["Chat Analyst", "Evaluation Dashboard"])

with chat_tab:
    left, right = st.columns([0.66, 0.34])

    with left:
        st.subheader("Ask a Business Question")
        selected = st.selectbox("Try a sample question", [""] + SAMPLE_QUESTIONS)
        typed_question = st.text_input("Question", value=selected, placeholder="Ask about revenue, reviews, sellers, delivery, freight, or cancellations")
        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

        if "chat_runs" not in st.session_state:
            st.session_state.chat_runs = []

        if run_clicked and typed_question.strip():
            effective_question = resolve_follow_up(typed_question.strip(), st.session_state.chat_runs)
            with st.spinner("Running analyst graph and applying business rules..."):
                final_state = run_agent(effective_question)
            final_state["display_question"] = typed_question.strip()
            st.session_state.chat_runs.insert(0, final_state)

        if st.session_state.chat_runs:
            latest = st.session_state.chat_runs[0]
            if latest.get("error"):
                st.error(latest["error"])

            st.markdown("#### Business Recommendation")
            render_recommendation(latest)

            st.markdown("#### Insight")
            st.write(latest.get("insight") or "No insight returned.")

            with st.expander("Why this SQL was used", expanded=True):
                for item in latest.get("sql_explanation", []):
                    st.write(f"- {item}")

            st.markdown("#### Generated SQL")
            st.code(latest.get("sql") or "", language="sql")

            st.markdown("#### Result")
            st.dataframe(result_dataframe(latest), use_container_width=True, hide_index=True)

            st.markdown("#### Chart")
            render_chart_from_state(latest)

            st.download_button(
                "Download Business Report",
                data=build_report(latest),
                file_name="business_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("Run a question to see the SQL, result table, recommendation, insight, and chart.")

    with right:
        st.subheader("Confidence & Safety")
        if st.session_state.get("chat_runs"):
            latest = st.session_state.chat_runs[0]
            safety = latest.get("safety_status") or {}
            st.metric("Difficulty", latest.get("difficulty_label") or "Unknown")
            st.metric("Rows Returned", latest.get("row_count", 0))
            st.metric("Recommendation Rule", latest.get("recommendation_rule") or "None")
            st.metric("Chart Type", latest.get("chart_type") or "None")
            st.markdown("#### KPIs Detected")
            st.write(", ".join(clean_label(k) for k in latest.get("detected_kpis", [])) or "None")
            st.markdown("#### Safety Status")
            st.json(safety)
            with st.expander("Session Memory"):
                for idx, run in enumerate(st.session_state.chat_runs[:5], start=1):
                    st.write(f"{idx}. {run.get('display_question') or run.get('question')}")
            with st.expander("Full latest state"):
                st.json({k: v for k, v in latest.items() if k != "rows"})
        else:
            st.write("No live run yet.")

with eval_tab:
    st.subheader("Saved Evaluation Results")
    eval_results = load_json(EVAL_RESULTS_PATH)
    faithfulness_results = load_json(FAITHFULNESS_RESULTS_PATH)

    if not eval_results:
        st.warning("eval_results.json was not found. Run python eval_harness.py first.")
    else:
        execution = eval_summary(eval_results)
        faithfulness = faithfulness_summary(faithfulness_results or []) if faithfulness_results else None

        metric_cols = st.columns(4)
        metric_cols[0].metric("Execution Accuracy", f"{execution['accuracy']:.0%}", f"{execution['passed']}/{execution['total']}")
        if faithfulness:
            metric_cols[1].metric("Faithful Cases", f"{faithfulness['case_score']:.0%}", f"{faithfulness['faithful']}/{faithfulness['total']}")
            metric_cols[2].metric("Numeric Grounding", f"{faithfulness['numeric_score']:.0%}", f"{faithfulness['grounded']}/{faithfulness['checked']}")
        else:
            metric_cols[1].metric("Faithful Cases", "Missing")
            metric_cols[2].metric("Numeric Grounding", "Missing")
        metric_cols[3].metric("Gold Questions", execution["total"])

        chart_cols = st.columns(3)
        with chart_cols[0]:
            st.markdown("#### Accuracy By Difficulty")
            diff_df = difficulty_dataframe(execution)
            if not diff_df.empty:
                st.bar_chart(diff_df, x="difficulty", y="accuracy")
                st.dataframe(diff_df, use_container_width=True, hide_index=True)

        with chart_cols[1]:
            st.markdown("#### Failure Types")
            fail_df = failure_dataframe(execution["failures"])
            if fail_df.empty:
                st.success("No execution failures in the saved file.")
            else:
                st.bar_chart(fail_df, x="failure_category", y="count")
                st.dataframe(fail_df, use_container_width=True, hide_index=True)

        with chart_cols[2]:
            st.markdown("#### Recommendation Rules")
            rec_df = recommendation_rule_dataframe(eval_results)
            if not rec_df.empty:
                st.bar_chart(rec_df, x="recommendation_rule", y="count")
                st.dataframe(rec_df, use_container_width=True, hide_index=True)

        st.markdown("#### Question-Level Results")
        table = merged_eval_table(eval_results, faithfulness_results)
        st.dataframe(table, use_container_width=True, hide_index=True)

        with st.expander("Raw result details"):
            st.json({"eval_results": eval_results, "faithfulness_results": faithfulness_results})



