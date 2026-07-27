import base64
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from agent_graph import build_graph
from db_adapter import get_database_backend, get_database_dialect

BASE_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_PATH = BASE_DIR / "eval_results.json"
FAITHFULNESS_RESULTS_PATH = BASE_DIR / "faithfulness_results.json"
HERO_IMAGE_PATH = BASE_DIR / "assets" / "analytics-command-center.png"

SAMPLE_QUESTIONS = [
    "What is the average review score for each payment type?",
    "Which sellers have the highest late delivery risk?",
    "Which sellers generated the highest total revenue?",
    "What is the average freight cost as a percentage of item price?",
    "How many orders were delivered later than their estimated delivery date?",
]

FOLLOW_UP_MARKERS = ["them", "those", "that", "these", "top 5", "only", "from them"]


THEME_CSS = """
<style>
:root {
  --bg: #f6f7fb;
  --panel: #ffffff;
  --ink: #172033;
  --muted: #667085;
  --line: #e5e7eb;
  --blue: #2563eb;
  --teal: #0f766e;
  --amber: #d97706;
  --red: #dc2626;
  --green: #15803d;
}
.stApp { background: var(--bg); color: var(--ink); }
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1400px; }
[data-testid="stHeader"] { background: rgba(246,247,251,0.85); backdrop-filter: blur(10px); }
[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stMetricValue"] { color: var(--ink); font-size: 1.45rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 18px;
}
.stTabs [aria-selected="true"] { border-color: var(--blue); color: var(--blue); }
.hero {
  position: relative;
  min-height: 360px;
  color: white;
  padding: 38px 42px;
  border-radius: 8px;
  margin-bottom: 18px;
  overflow: hidden;
  background-size: cover;
  background-position: center;
  box-shadow: 0 24px 48px rgba(15,23,42,.24);
}
.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(15,23,42,.88) 0%, rgba(15,23,42,.62) 42%, rgba(15,23,42,.10) 100%);
}
.hero-content { position: relative; z-index: 1; max-width: 720px; }
.hero-kicker { margin: 0 0 10px 0; color: #99f6e4; font-size: .82rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.hero h1 { margin: 0 0 12px 0; font-size: 2.35rem; line-height: 1.08; letter-spacing: 0; max-width: 680px; }
.hero p { margin: 0; color: #e0f2fe; font-size: 1.02rem; max-width: 640px; }
.hero-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
.hero-pill {
  display: inline-flex;
  align-items: center;
  min-height: 32px;
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,.12);
  color: #ffffff;
  border: 1px solid rgba(255,255,255,.24);
  backdrop-filter: blur(10px);
  font-size: .84rem;
  font-weight: 700;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
  margin-bottom: 14px;
}
.panel h3 { margin: 0 0 10px 0; font-size: 1.05rem; letter-spacing: 0; }
.callout {
  border-radius: 8px;
  padding: 16px 18px;
  border: 1px solid var(--line);
  background: #ffffff;
  margin-bottom: 14px;
}
.callout.high { border-left: 5px solid var(--red); background: #fff7f7; }
.callout.medium { border-left: 5px solid var(--amber); background: #fffbeb; }
.callout.low { border-left: 5px solid var(--green); background: #f0fdf4; }
.callout h3 { margin: 0 0 6px 0; font-size: 1.1rem; letter-spacing: 0; }
.callout p { margin: 4px 0; color: var(--ink); }
.badge-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.badge {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 4px 10px;
  border-radius: 999px;
  background: #eef2ff;
  color: #3730a3;
  border: 1px solid #c7d2fe;
  font-size: .82rem;
  font-weight: 600;
}
.badge.teal { background: #ecfdf5; color: #0f766e; border-color: #99f6e4; }
.badge.amber { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.section-title { font-size: 1.05rem; font-weight: 700; margin: 18px 0 8px 0; color: var(--ink); }
.small-muted { color: var(--muted); font-size: .9rem; }
hr.soft { border: none; border-top: 1px solid var(--line); margin: 14px 0; }
</style>
"""


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

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
            "recommendation_rule": row.get("recommendation_rule") or "",
            "priority": row.get("priority") or "",
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
{state.get('display_question') or state.get('question', '')}

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


def render_badges(items: list[str], style: str = ""):
    if not items:
        st.markdown('<span class="small-muted">None detected</span>', unsafe_allow_html=True)
        return
    badges = "".join(f'<span class="badge {style}">{clean_label(item)}</span>' for item in items)
    st.markdown(f'<div class="badge-row">{badges}</div>', unsafe_allow_html=True)


def render_recommendation(state: dict):
    priority = state.get("priority") or "Low"
    level = priority.lower() if priority in {"High", "Medium", "Low"} else "low"
    st.markdown(
        f"""
        <div class="callout {level}">
          <h3>{state.get('recommendation') or 'Monitor KPI'} <span class="small-muted">{priority} priority</span></h3>
          <p>{state.get('recommendation_reason') or 'No rule reason returned.'}</p>
          <hr class="soft" />
          <p><strong>Recommended action:</strong> {state.get('recommended_action') or 'No action returned.'}</p>
          <div class="badge-row"><span class="badge amber">{clean_label(state.get('recommendation_rule') or 'monitoring_rule')}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str):
    st.markdown(f'<div class="panel"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)


st.set_page_config(page_title="Autonomous Data Analyst Agent", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
hero_uri = image_data_uri(HERO_IMAGE_PATH)
st.markdown(
    f"""
    <div class="hero" style="background-image: url('{hero_uri}')">
      <div class="hero-content">
        <p class="hero-kicker">AI Data Analyst + Decision Intelligence</p>
        <h1>Ask business questions. Get validated SQL, grounded insights, and recommended actions.</h1>
        <p>A production-minded analytics app for e-commerce data: schema-aware SQL generation, safety checks, interactive charts, faithfulness evaluation, and Azure-ready database support.</p>
        <div class="hero-actions">
          <span class="hero-pill">LangGraph agents</span>
          <span class="hero-pill">SQL validation</span>
          <span class="hero-pill">Business rules</span>
          <span class="hero-pill">Azure-ready</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

status_cols = st.columns(2)
status_cols[0].caption(f"Database backend: {get_database_backend()}")
status_cols[1].caption(f"SQL dialect: {get_database_dialect()}")

chat_tab, eval_tab = st.tabs(["Chat Analyst", "Evaluation Dashboard"])

with chat_tab:
    left, right = st.columns([0.67, 0.33], gap="large")

    with left:
        st.markdown('<div class="section-title">Business Question</div>', unsafe_allow_html=True)
        selected = st.selectbox("Sample question", [""] + SAMPLE_QUESTIONS, label_visibility="collapsed")
        typed_question = st.text_input(
            "Question",
            value=selected,
            placeholder="Ask about revenue, reviews, sellers, delivery, freight, or cancellations",
            label_visibility="collapsed",
        )
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

            top_metrics = st.columns(4)
            top_metrics[0].metric("Rows", latest.get("row_count", 0))
            top_metrics[1].metric("Difficulty", latest.get("difficulty_label") or "Unknown")
            top_metrics[2].metric("Priority", latest.get("priority") or "Low")
            top_metrics[3].metric("Chart", clean_label(latest.get("chart_type") or "None"))

            st.markdown('<div class="section-title">Business Recommendation</div>', unsafe_allow_html=True)
            render_recommendation(latest)

            st.markdown('<div class="section-title">Insight</div>', unsafe_allow_html=True)
            render_panel("Grounded narrative", latest.get("insight") or "No insight returned.")

            st.markdown('<div class="section-title">Why This Query Was Used</div>', unsafe_allow_html=True)
            with st.container(border=True):
                for item in latest.get("sql_explanation", []):
                    st.write(f"- {item}")

            st.markdown('<div class="section-title">Generated SQL</div>', unsafe_allow_html=True)
            st.code(latest.get("sql") or "", language="sql")

            st.markdown('<div class="section-title">Result Table</div>', unsafe_allow_html=True)
            st.dataframe(result_dataframe(latest), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Chart</div>', unsafe_allow_html=True)
            render_chart_from_state(latest)

            st.download_button(
                "Download Business Report",
                data=build_report(latest),
                file_name="business_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            render_panel("Ready", "Ask a business question to generate SQL, run the analysis, review the recommendation, and export the report.")

    with right:
        st.markdown('<div class="section-title">Confidence & Safety</div>', unsafe_allow_html=True)
        if st.session_state.get("chat_runs"):
            latest = st.session_state.chat_runs[0]
            safety = latest.get("safety_status") or {}
            with st.container(border=True):
                st.metric("SQL Validation", safety.get("sql_validation", "Passed"))
                st.metric("Rule Fired", clean_label(latest.get("recommendation_rule") or "None"))
                st.metric("Plan", clean_label(latest.get("plan") or "Unknown"))
                st.markdown("**KPIs detected**")
                render_badges(latest.get("detected_kpis", []), "teal")
                st.markdown("**Tables used**")
                render_badges(safety.get("tables_used", []))

            with st.expander("Session Memory"):
                for idx, run in enumerate(st.session_state.chat_runs[:5], start=1):
                    st.write(f"{idx}. {run.get('display_question') or run.get('question')}")
            with st.expander("Full latest state"):
                st.json({k: v for k, v in latest.items() if k != "rows"})
        else:
            render_panel("No run yet", "The safety panel will populate after the first analysis.")

with eval_tab:
    st.markdown('<div class="section-title">Evaluation Overview</div>', unsafe_allow_html=True)
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

        chart_cols = st.columns(3, gap="large")
        with chart_cols[0]:
            st.markdown('<div class="section-title">Accuracy By Difficulty</div>', unsafe_allow_html=True)
            diff_df = difficulty_dataframe(execution)
            if not diff_df.empty:
                st.bar_chart(diff_df, x="difficulty", y="accuracy")
                st.dataframe(diff_df, use_container_width=True, hide_index=True)

        with chart_cols[1]:
            st.markdown('<div class="section-title">Failure Types</div>', unsafe_allow_html=True)
            fail_df = failure_dataframe(execution["failures"])
            if fail_df.empty:
                st.success("No execution failures in the saved file.")
            else:
                st.bar_chart(fail_df, x="failure_category", y="count")
                st.dataframe(fail_df, use_container_width=True, hide_index=True)

        with chart_cols[2]:
            st.markdown('<div class="section-title">Recommendation Rules</div>', unsafe_allow_html=True)
            rec_df = recommendation_rule_dataframe(eval_results)
            if not rec_df.empty:
                st.bar_chart(rec_df, x="recommendation_rule", y="count")
                st.dataframe(rec_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Question-Level Results</div>', unsafe_allow_html=True)
        table = merged_eval_table(eval_results, faithfulness_results)
        st.dataframe(table, use_container_width=True, hide_index=True)

        with st.expander("Raw result details"):
            st.json({"eval_results": eval_results, "faithfulness_results": faithfulness_results})



