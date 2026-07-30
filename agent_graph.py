import os
import re
import time
import json
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, END
import plotly.express as px
import pandas as pd
from business_rules import evaluate_business_rules
from db_adapter import execute_query, get_database_dialect, get_schema_objects, get_schema_text, validate_sql

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))



# ============================================================
# SHARED STATE -- this is what flows between every node
# ============================================================
class AgentState(TypedDict):
    question: str
    plan: str                  # router's decision: "single_query" or "multi_step"
    sql: str
    validation_ok: bool
    validation_message: str
    columns: List[str]
    rows: List[tuple]
    row_count: int
    insight: str
    detected_kpis: List[str]
    difficulty_label: str
    sql_explanation: List[str]
    recommendations: List[dict]
    recommendation: str
    priority: str
    recommendation_rule: str
    recommendation_reason: str
    recommended_action: str
    safety_status: dict
    chart_path: Optional[str]
    chart_type: Optional[str]
    error: Optional[str]


# ============================================================
# SHARED HELPER: retry-safe Gemini call
# ============================================================
def call_gemini_with_retry(prompt: str, max_retries: int = 5) -> str:
    last_error = None
    models_to_try = [
        "gemini-flash-latest",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-flash-lite-latest",
        "gemini-flash-lite-latest",
    ]
    for attempt in range(max_retries):
        model = models_to_try[min(attempt, len(models_to_try) - 1)]
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text.strip()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  ({model} busy, retrying in {wait}s... attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
    raise last_error


def get_actual_schema_objects():
    return get_schema_objects()


# ============================================================
# NODE 1: ROUTER
# Decides whether the question needs one query or a breakdown.
# For now, multi-step questions are flagged but still attempted
# as a single query -- true decomposition is a future extension.
# ============================================================
def router_node(state: AgentState) -> AgentState:
    print(f"\n[ROUTER] Analyzing question: {state['question']}")

    prompt = f"""Classify this business question about an e-commerce database as either:
- "single_query" if it can be answered with one SQL query
- "multi_step" if it truly requires multiple separate queries whose results feed into each other

Question: {state['question']}

Respond with ONLY the single word: single_query OR multi_step"""

    plan = call_gemini_with_retry(prompt).strip().lower()
    if "multi" in plan:
        plan = "multi_step"
    else:
        plan = "single_query"

    print(f"[ROUTER] Plan: {plan}")
    state["plan"] = plan
    return state


# ============================================================
# NODE 2: SQL AGENT
# Generates SQL, validates it against the real schema before
# anything is executed (catches hallucinated tables/columns).
# ============================================================
def sql_agent_node(state: AgentState) -> AgentState:
    print(f"\n[SQL AGENT] Generating SQL...")

    dialect = get_database_dialect()
    schema_text = get_schema_text()

    prompt = f"""You are a SQL expert. Given the database schema below, write a single valid {dialect} query to answer the question.

{schema_text}

Question: {state['question']}

Rules:
- Return ONLY the SQL query, no explanation, no markdown formatting, no ```sql fences.
- Use only tables and columns listed above.
- Use clean, business-friendly column aliases for calculated fields, such as avg_review_score or total_revenue; do not expose raw expressions like AVG(t1.review_score) as result column names.
- End the query with a semicolon.
"""
    sql = call_gemini_with_retry(prompt)
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.MULTILINE).strip()
    print(f"[SQL AGENT] Generated:\n{sql}")

    validation_ok, validation_message = validate_sql(sql)
    print(f"[SQL AGENT] Validation: {validation_message}")

    state["sql"] = sql
    state["validation_ok"] = validation_ok
    state["validation_message"] = validation_message
    if not validation_ok:
        state["error"] = f"SQL validation failed: {validation_message}"
    return state

def validation_router(state: AgentState) -> str:
    """Conditional edge: skip straight to end if SQL is invalid."""
    return "execution_agent" if state["validation_ok"] else "end_with_error"


# ============================================================
# NODE 3: EXECUTION AGENT
# Runs the validated query, returns results + basic stats.
# ============================================================
def execution_agent_node(state: AgentState) -> AgentState:
    print(f"\n[EXECUTION AGENT] Running query...")
    try:
        columns, rows = execute_query(state["sql"])
        state["columns"] = columns
        state["rows"] = rows
        state["row_count"] = len(rows)
        print(f"[EXECUTION AGENT] Got {len(rows)} row(s), columns: {columns}")
    except Exception as e:
        state["error"] = f"SQL execution failed: {e}"
        state["rows"] = []
        state["columns"] = []
        state["row_count"] = 0
        print(f"[EXECUTION AGENT] ERROR: {e}")
    return state

# ============================================================
# NODE 4: INSIGHT AGENT
# Turns raw numbers into a written narrative, strictly grounded
# in the actual returned data (we pass the real rows into the
# prompt so the model can't invent numbers not present).
# ============================================================
def insight_agent_node(state: AgentState) -> AgentState:
    print(f"\n[INSIGHT AGENT] Writing narrative...")

    if state.get("error"):
        state["insight"] = f"Could not generate insight due to an earlier error: {state['error']}"
        return state

    data_preview = list(zip(state["columns"], state["rows"][0])) if state["rows"] else []

    prompt = f"""You are a data analyst. Write a short (2-4 sentence) business insight
answering the question below, using ONLY the data provided. Do not invent any
numbers that are not present in the data. If the data doesn't fully answer the
question, say so honestly.

Question: {state['question']}

Result columns: {state['columns']}
Result rows (all of them): {state['rows']}

Write the insight now:"""

    insight = call_gemini_with_retry(prompt)
    print(f"[INSIGHT AGENT] {insight}")
    state["insight"] = insight
    return state


# ============================================================
# NODE 5: BUSINESS RULES / RECOMMENDATION LAYER
# Converts analytical facts into deterministic business actions.
# ============================================================
def recommendation_node(state: AgentState) -> AgentState:
    print(f"\n[RECOMMENDATION] Applying business rules...")

    if state.get("error"):
        state["detected_kpis"] = []
        state["difficulty_label"] = "Unknown"
        state["sql_explanation"] = []
        state["recommendations"] = []
        state["recommendation"] = "No recommendation"
        state["priority"] = "None"
        state["recommendation_rule"] = "not_applicable"
        state["recommendation_reason"] = state.get("error") or "Earlier pipeline error."
        state["recommended_action"] = "Fix the pipeline error before making a business recommendation."
        state["safety_status"] = {
            "sql_validation": "Failed",
            "tables_used": [],
            "rows_returned": state.get("row_count", 0),
            "recommendation_rule": "not_applicable",
            "insight_grounding": "Not checked",
        }
        return state

    decision = evaluate_business_rules(
        question=state["question"],
        columns=state.get("columns", []),
        rows=state.get("rows", []),
        sql=state.get("sql", ""),
    )
    state.update(decision)
    print(f"[RECOMMENDATION] {state['recommendation']} ({state['priority']}) via {state['recommendation_rule']}")
    return state

# ============================================================
# NODE 6: CHART AGENT
# Picks a chart type based on the shape of the result and saves
# an HTML chart (no extra image-rendering dependency needed).
# ============================================================
def chart_agent_node(state: AgentState) -> AgentState:
    print(f"\n[CHART AGENT] Building chart...")

    if state.get("error") or not state["rows"]:
        state["chart_path"] = None
        state["chart_type"] = None
        print("[CHART AGENT] Skipped -- no data to chart.")
        return state

    df = pd.DataFrame(state["rows"], columns=state["columns"])
    df.columns = [c.replace("_", " ").replace("(", "").replace(")", "").title() for c in df.columns]

    # Convert date-like text columns so trend questions become line charts.
    for col in df.columns:
        col_lower = col.lower()
        if any(token in col_lower for token in ["date", "month", "year", "time"]):
            converted = pd.to_datetime(df[col], errors="coerce")
            if converted.notna().sum() >= max(1, len(df) // 2):
                df[col] = converted

    num_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]

    os.makedirs("charts", exist_ok=True)
    chart_path = os.path.join("charts", "latest_chart.html")
    title = state.get("display_question") or state.get("question") or "Analysis Result"

    chart_palette = ["#2563eb", "#0f766e", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#65a30d", "#ea580c"]
    question_text = str(title).lower()

    def has_any(tokens: list[str]) -> bool:
        return any(token in question_text for token in tokens)

    pie_requested = has_any(["pie", "donut", "percentage", "percent", "share", "proportion", "composition", "split", "breakdown"])
    line_requested = has_any(["trend", "over time", "timeline", "monthly", "daily", "weekly", "yearly", "by month", "by date", "line graph", "line chart"])
    scatter_requested = has_any(["scatter", "relationship", "correlation", " vs ", " versus ", "against", "compare relationship"])
    histogram_requested = has_any(["histogram", "frequency", "spread", "distribution of"])
    rank_requested = has_any(["top", "highest", "lowest", "rank", "ranking", "compare", "comparison", "by "])

    def preferred_numeric(columns: list[str]) -> str:
        priority_terms = ["percent", "percentage", "share", "count", "total", "sum", "avg", "average", "rate", "score", "revenue", "cost"]
        return next((col for col in columns if any(term in col.lower() for term in priority_terms)), columns[0])

    try:
        if len(df) == 1 and num_cols:
            metric_df = df[num_cols].T.reset_index()
            metric_df.columns = ["Metric", "Value"]
            fig = px.bar(
                metric_df,
                x="Metric",
                y="Value",
                title="KPI Snapshot",
                text="Value",
                labels={"Metric": "Metric", "Value": "Value"},
                color_discrete_sequence=chart_palette,
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            chart_type = "kpi_bar"

        elif date_cols and num_cols:
            x_col = date_cols[0]
            y_cols = num_cols[:3]
            trend_df = df.sort_values(x_col)
            fig = px.line(
                trend_df,
                x=x_col,
                y=y_cols,
                markers=True,
                title=title,
                labels={"value": "Value", "variable": "Metric"},
                color_discrete_sequence=chart_palette,
            )
            chart_type = "line"

        elif scatter_requested and len(num_cols) >= 2:
            color_col = cat_cols[0] if cat_cols and len(df) <= 40 else None
            fig = px.scatter(
                df,
                x=num_cols[0],
                y=num_cols[1],
                size=num_cols[2] if len(num_cols) >= 3 else None,
                color=color_col,
                title=title,
                hover_data=df.columns,
                color_discrete_sequence=chart_palette,
            )
            chart_type = "scatter"

        elif pie_requested and cat_cols and num_cols and 2 <= len(df) <= 25:
            x_col = cat_cols[0]
            y_col = preferred_numeric(num_cols)
            plot_df = df.sort_values(y_col, ascending=False).head(15)
            fig = px.pie(
                plot_df,
                names=x_col,
                values=y_col,
                title=f"{y_col} Distribution by {x_col}",
                color_discrete_sequence=chart_palette,
                hole=0.35,
            )
            fig.update_traces(
                textinfo="percent+label",
                textposition="inside",
                insidetextfont=dict(color="#ffffff", size=13),
                outsidetextfont=dict(color="#172033", size=12),
                marker=dict(line=dict(color="#ffffff", width=2)),
            )
            chart_type = "pie"

        elif line_requested and cat_cols and num_cols and len(df) <= 60:
            x_col = cat_cols[0]
            y_cols = num_cols[:3]
            fig = px.line(
                df,
                x=x_col,
                y=y_cols,
                markers=True,
                title=title,
                labels={"value": "Value", "variable": "Metric"},
                color_discrete_sequence=chart_palette,
            )
            chart_type = "line"

        elif len(num_cols) >= 2 and not cat_cols:
            if histogram_requested and not scatter_requested:
                fig = px.histogram(
                    df,
                    x=num_cols[0],
                    nbins=min(30, max(8, len(df) // 2)),
                    title=f"Distribution of {num_cols[0]}",
                    color_discrete_sequence=chart_palette,
                )
                fig.update_traces(marker_color="#2563eb")
                chart_type = "histogram"
            else:
                fig = px.scatter(
                    df,
                    x=num_cols[0],
                    y=num_cols[1],
                    size=num_cols[2] if len(num_cols) >= 3 else None,
                    title=title,
                    hover_data=df.columns,
                    color_discrete_sequence=chart_palette,
                )
                chart_type = "scatter"

        elif len(num_cols) >= 2 and cat_cols:
            x_col = cat_cols[0]
            y_cols = num_cols[:3]
            plot_df = df.sort_values(y_cols[0], ascending=False).head(15)
            fig = px.bar(
                plot_df,
                x=x_col,
                y=y_cols,
                barmode="group",
                title=title,
                labels={"value": "Value", "variable": "Metric"},
                color_discrete_sequence=chart_palette,
            )
            chart_type = "grouped_bar"

        elif cat_cols and num_cols:
            x_col = cat_cols[0]
            y_col = preferred_numeric(num_cols)
            plot_df = df.sort_values(y_col, ascending=True).tail(15)
            fig = px.bar(
                plot_df,
                x=y_col,
                y=x_col,
                orientation="h",
                title=f"{y_col} by {x_col}",
                labels={x_col: x_col, y_col: y_col},
                color_discrete_sequence=chart_palette,
            )
            fig.update_traces(marker_color="#2563eb")
            chart_type = "horizontal_bar"

        elif num_cols and len(df) > 1:
            fig = px.histogram(
                df,
                x=num_cols[0],
                nbins=min(30, max(8, len(df) // 2)),
                title=f"Distribution of {num_cols[0]}",
                color_discrete_sequence=chart_palette,
            )
            fig.update_traces(marker_color="#2563eb")
            chart_type = "histogram"

        else:
            state["chart_path"] = None
            state["chart_type"] = None
            print("[CHART AGENT] Result shape not chartable -- skipped.")
            return state

        fig.update_layout(
            template="plotly_white",
            height=520,
            margin=dict(l=60, r=30, t=70, b=70),
            legend_title_text="Metric",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#172033"),
            title_font=dict(color="#172033", size=20),
            hoverlabel=dict(bgcolor="#ffffff", font_color="#172033", bordercolor="#cbd5e1"),
        )
        fig.update_xaxes(gridcolor="#e5e7eb", zerolinecolor="#cbd5e1", title_font=dict(color="#172033"), tickfont=dict(color="#172033"))
        fig.update_yaxes(gridcolor="#eef2f7", zerolinecolor="#cbd5e1", title_font=dict(color="#172033"), tickfont=dict(color="#172033"))
        fig.write_html(chart_path)
        state["chart_path"] = chart_path
        state["chart_type"] = chart_type
        print(f"[CHART AGENT] Saved {chart_type} chart to {chart_path}")
    except Exception as e:
        state["chart_path"] = None
        state["chart_type"] = None
        print(f"[CHART AGENT] Chart generation failed: {e}")

    return state
def end_with_error_node(state: AgentState) -> AgentState:
    print(f"\n[STOPPED] {state.get('error', 'Unknown error')}")
    return state


# ============================================================
# BUILD THE GRAPH
# ============================================================
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("execution_agent", execution_agent_node)
    graph.add_node("insight_agent", insight_agent_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("chart_agent", chart_agent_node)
    graph.add_node("end_with_error", end_with_error_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "sql_agent")
    graph.add_conditional_edges(
        "sql_agent",
        validation_router,
        {"execution_agent": "execution_agent", "end_with_error": "end_with_error"}
    )
    graph.add_edge("execution_agent", "insight_agent")
    graph.add_edge("insight_agent", "recommendation")
    graph.add_edge("recommendation", "chart_agent")
    graph.add_edge("chart_agent", END)
    graph.add_edge("end_with_error", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state: AgentState = {
        "question": "What is the average review score for each payment type?",
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

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Question: {final_state['question']}")
    print(f"SQL: {final_state['sql']}")
    print(f"Insight: {final_state['insight']}")
    print(f"Recommendation: {final_state.get('recommendation')} ({final_state.get('priority')})")
    print(f"Action: {final_state.get('recommended_action')}")
    print(f"Chart: {final_state['chart_path']}")

