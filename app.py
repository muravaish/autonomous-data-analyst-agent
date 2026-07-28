import base64
import json
import os
import re
import sqlite3
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
UPLOAD_DB_PATH = BASE_DIR / "session_uploads" / "uploaded_data.db"

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
/* Keep Streamlit controls in light theme, including hover/focus states. */
.stTextInput input,
.stTextInput input:hover,
.stTextInput input:focus,
[data-baseweb="input"] input,
[data-baseweb="input"] input:hover,
[data-baseweb="input"] input:focus {
  background: #ffffff !important;
  color: var(--ink) !important;
  border-color: #cbd5e1 !important;
  box-shadow: none !important;
}
.stTextInput [data-baseweb="base-input"],
.stTextInput [data-baseweb="base-input"]:hover,
.stTextInput [data-baseweb="base-input"]:focus-within {
  background: #ffffff !important;
  border-color: #cbd5e1 !important;
}
.stButton > button {
  background: var(--blue) !important;
  color: #ffffff !important;
  border: 1px solid var(--blue) !important;
  border-radius: 8px !important;
  min-height: 46px;
  font-weight: 700 !important;
}
.stButton > button:hover,
.stButton > button:focus {
  background: #1d4ed8 !important;
  color: #ffffff !important;
  border-color: #1d4ed8 !important;
}
.stDownloadButton > button,
.stDownloadButton > button:hover,
.stDownloadButton > button:focus {
  background: #ffffff !important;
  color: var(--blue) !important;
  border: 1px solid #bfdbfe !important;
  border-radius: 8px !important;
  min-height: 44px;
  font-weight: 700 !important;
  box-shadow: 0 1px 2px rgba(16,24,40,.04) !important;
}
.stDownloadButton > button:hover,
.stDownloadButton > button:focus {
  background: #eff6ff !important;
  border-color: var(--blue) !important;
}
[data-testid="stFileUploader"] {
  background: #ffffff !important;
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  padding: 12px 14px !important;
  box-shadow: 0 1px 2px rgba(16,24,40,.04) !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section:hover {
  background: #f8fafc !important;
  border: 1px dashed #cbd5e1 !important;
  border-radius: 8px !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] button:focus {
  background: #ffffff !important;
  color: var(--blue) !important;
  border: 1px solid #bfdbfe !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
  color: var(--muted) !important;
}
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary:hover {
  background: #ffffff !important;
  color: var(--ink) !important;
  border-color: var(--line) !important;
}
.sample-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin: 10px 0 12px 0;
}
.sample-card {
  display: block;
  min-height: 44px;
  padding: 10px 12px;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  background: #f8fafc;
  color: #334155;
  font-size: .86rem;
  font-weight: 650;
}
.sample-card:hover { border-color: var(--blue); color: var(--blue); background: #eff6ff; }
.hero {
  position: relative;
  min-height: 310px;
  color: white;
  padding: 34px 38px;
  border-radius: 8px;
  margin-bottom: 14px;
  overflow: hidden;
  background-size: cover;
  background-position: center;
  box-shadow: 0 24px 48px rgba(15,23,42,.22);
}
.hero::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(12,18,32,.86) 0%, rgba(12,18,32,.54) 40%, rgba(12,18,32,.06) 100%);
}
.hero-content { position: relative; z-index: 1; max-width: 600px; }
.hero-kicker { margin: 0 0 10px 0; color: #99f6e4; font-size: .78rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.hero h1 { margin: 0 0 12px 0; font-size: 2.55rem; line-height: 1.02; letter-spacing: 0; max-width: 560px; }
.hero p { margin: 0; color: #e0f2fe; font-size: 1rem; max-width: 500px; }
.hero-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
.hero-pill {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 5px 11px;
  border-radius: 999px;
  background: rgba(255,255,255,.13);
  color: #ffffff;
  border: 1px solid rgba(255,255,255,.25);
  backdrop-filter: blur(10px);
  font-size: .8rem;
  font-weight: 700;
}
.status-strip {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 0 0 16px 0;
}
.status-chip {
  display: inline-flex;
  gap: 7px;
  align-items: center;
  min-height: 34px;
  padding: 6px 12px;
  border-radius: 999px;
  background: #ffffff;
  border: 1px solid var(--line);
  color: var(--muted);
  font-size: .84rem;
  font-weight: 650;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--teal); }
.question-shell {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 12px 28px rgba(16,24,40,.06);
  margin-bottom: 14px;
}
.quick-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 10px 0 16px 0;
}
.quick-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  min-height: 92px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.quick-card strong { display: block; color: var(--ink); font-size: .95rem; margin-bottom: 5px; }
.quick-card span { color: var(--muted); font-size: .84rem; }
.empty-state {
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  border: 1px dashed #cbd5e1;
  border-radius: 8px;
  padding: 22px;
  margin-top: 12px;
}
.empty-state h3 { margin: 0 0 6px 0; font-size: 1.15rem; color: var(--ink); }
.empty-state p { margin: 0; color: var(--muted); }
.history-list {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px 14px;
  margin-top: 12px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.history-list h3 { margin: 0 0 8px 0; font-size: .98rem; color: var(--ink); }
.history-item {
  display: block;
  padding: 8px 0;
  border-top: 1px solid #eef2f7;
  color: var(--muted);
  font-size: .86rem;
  line-height: 1.35;
}
.history-item:first-of-type { border-top: none; }
.sql-panel {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px;
  box-shadow: 0 1px 2px rgba(16,24,40,.04);
}
.sql-panel h3 { margin: 0 0 10px 0; font-size: 1rem; color: var(--ink); }
.sql-code {
  display: block;
  white-space: pre-wrap;
  word-break: break-word;
  background: #f8fafc;
  color: #0f172a;
  border: 1px solid #dbe3ef;
  border-radius: 8px;
  padding: 14px;
  font-family: Consolas, "Courier New", monospace;
  font-size: .9rem;
  line-height: 1.55;
}
.sql-explain {
  margin-top: 12px;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px 14px;
  color: var(--muted);
}
.sql-explain strong { color: var(--ink); }
.sql-explain ul { margin: 8px 0 0 18px; padding: 0; }
.sql-explain li { margin-bottom: 6px; }
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

def safe_sql_name(value: str, fallback: str = "uploaded_data") -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value).strip().lower()).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    columns = []
    for idx, column in enumerate(df.columns):
        base = safe_sql_name(column, f"column_{idx + 1}")
        count = seen.get(base, 0)
        seen[base] = count + 1
        columns.append(base if count == 0 else f"{base}_{count + 1}")
    df = df.copy()
    df.columns = columns
    return df


def activate_default_database() -> None:
    os.environ["DATABASE_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(BASE_DIR / "olist.db")


def activate_uploaded_database(uploaded_file) -> dict:
    UPLOAD_DB_PATH.parent.mkdir(exist_ok=True)
    df = pd.read_csv(uploaded_file)
    df = clean_dataframe_columns(df)
    table_name = "uploaded_data"

    with sqlite3.connect(UPLOAD_DB_PATH) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    os.environ["DATABASE_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = str(UPLOAD_DB_PATH)
    return {
        "name": uploaded_file.name,
        "table": table_name,
        "rows": len(df),
        "columns": list(df.columns),
        "db_path": str(UPLOAD_DB_PATH),
    }

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
if st.session_state.get("dataset_info"):
    os.environ["DATABASE_BACKEND"] = "sqlite"
    os.environ["SQLITE_DB_PATH"] = st.session_state.dataset_info["db_path"]
else:
    activate_default_database()

st.markdown(THEME_CSS, unsafe_allow_html=True)
hero_uri = image_data_uri(HERO_IMAGE_PATH)
st.markdown(
    f"""
    <div class="hero" style="background-image: url('{hero_uri}')">
      <div class="hero-content">
        <p class="hero-kicker">AI Commerce Analyst</p>
        <h1>Answers you can verify.</h1>
        <p>Ask a question. Get SQL, insight, chart, and action.</p>
        <div class="hero-actions">
          <span class="hero-pill">Validated SQL</span>
          <span class="hero-pill">Faithfulness checks</span>
          <span class="hero-pill">Azure-ready</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status-strip">
      <span class="status-chip"><span class="status-dot"></span>{get_database_backend()}</span>
      <span class="status-chip"><span class="status-dot"></span>{get_database_dialect()}</span>
      <span class="status-chip"><span class="status-dot"></span>20 gold checks</span>

    </div>
    """,
    unsafe_allow_html=True,
)

chat_tab, eval_tab = st.tabs(["Chat Analyst", "Evaluation Dashboard"])

with chat_tab:
    left, right = st.columns([0.67, 0.33], gap="large")

    with left:
        st.markdown(
            """
            <div class="question-shell">
              <div class="section-title" style="margin-top:0">Ask The Analyst</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            signature = f"{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.get("upload_signature") != signature:
                try:
                    st.session_state.dataset_info = activate_uploaded_database(uploaded_file)
                    st.session_state.upload_signature = signature
                    st.session_state.chat_runs = []
                    st.session_state.draft_question = ""
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load CSV: {exc}")
            else:
                os.environ["DATABASE_BACKEND"] = "sqlite"
                os.environ["SQLITE_DB_PATH"] = st.session_state.dataset_info["db_path"]

        dataset_info = st.session_state.get("dataset_info")
        if dataset_info:
            st.caption(f"Dataset: {dataset_info['name']} | table `{dataset_info['table']}` | {dataset_info['rows']} rows")
            if st.button("Use Olist sample", use_container_width=True):
                st.session_state.pop("dataset_info", None)
                st.session_state.pop("upload_signature", None)
                st.session_state.chat_runs = []
                activate_default_database()
                st.rerun()
        else:
            st.caption("Dataset: Olist e-commerce sample")

        if "draft_question" not in st.session_state:
            st.session_state.draft_question = ""

        typed_question = st.text_input(
            "Question",
            key="draft_question",
            placeholder="Example: Which sellers have the highest late delivery risk?",
            label_visibility="collapsed",
        )

        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

        if "chat_runs" not in st.session_state:
            st.session_state.chat_runs = []

        if run_clicked and typed_question.strip():
            effective_question = resolve_follow_up(typed_question.strip(), st.session_state.chat_runs)
            with st.spinner("Running analysis..."):
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

            answer_tab, chart_tab, data_tab, sql_tab = st.tabs(["Answer", "Chart", "Data", "SQL"])

            with answer_tab:
                render_recommendation(latest)
                render_panel("Insight", latest.get("insight") or "No insight returned.")
                st.download_button(
                    "Download Report",
                    data=build_report(latest),
                    file_name="business_analysis_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            with chart_tab:
                render_chart_from_state(latest)

            with data_tab:
                st.dataframe(result_dataframe(latest), use_container_width=True, hide_index=True)

            with sql_tab:
                sql_text = latest.get("sql") or "No SQL returned."
                explanation_items = "".join(f"<li>{item}</li>" for item in latest.get("sql_explanation", []))
                if not explanation_items:
                    explanation_items = "<li>No explanation returned.</li>"
                st.markdown(
                    f"""
                    <div class="sql-panel">
                      <h3>Generated SQL</h3>
                      <code class="sql-code">{sql_text}</code>
                      <div class="sql-explain"><strong>Why this query</strong><ul>{explanation_items}</ul></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="quick-grid">
                  <div class="quick-card"><strong>Seller risk</strong><span>Which sellers have the highest late delivery risk?</span></div>
                  <div class="quick-card"><strong>Revenue</strong><span>Which sellers generated the highest total revenue?</span></div>
                  <div class="quick-card"><strong>Reviews</strong><span>What is the average review score by payment type?</span></div>
                </div>
                <div class="empty-state"><h3>Ready for analysis</h3><p>Type one business question above and run the analyst.</p></div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)
        if st.session_state.get("chat_runs"):
            latest = st.session_state.chat_runs[0]
            safety = latest.get("safety_status") or {}
            with st.container(border=True):
                st.metric("SQL", safety.get("sql_validation", "Passed"))
                st.metric("Rule", clean_label(latest.get("recommendation_rule") or "None"))
                st.metric("Plan", clean_label(latest.get("plan") or "Unknown"))
                st.markdown("**KPIs**")
                render_badges(latest.get("detected_kpis", []), "teal")
                st.markdown("**Tables**")
                render_badges(safety.get("tables_used", []))

            history_items = "".join(
                f"<span class='history-item'>{idx}. {run.get('display_question') or run.get('question')}</span>"
                for idx, run in enumerate(st.session_state.chat_runs[:5], start=1)
            )
            st.markdown(
                f"<div class='history-list'><h3>Recent Questions</h3>{history_items}</div>",
                unsafe_allow_html=True,
            )
        else:
            render_panel("Standing by", "Run a question to see validation, KPIs, and tables.")

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








