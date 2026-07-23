import os
import re
import time
import sqlite3
import json
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from google import genai
from langgraph.graph import StateGraph, END
import plotly.express as px
import pandas as pd
from business_rules import evaluate_business_rules

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

DB_PATH = "olist.db"

SCHEMA = """
Tables and columns:

customers(customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state)
orders(order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date)
order_items(order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value)
order_payments(order_id, payment_sequential, payment_type, payment_installments, payment_value)
order_reviews(review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp)
products(product_id, product_category_name, product_name_lenght, product_description_lenght, product_photos_qty, product_weight_g, product_length_cm, product_height_cm, product_width_cm)
sellers(seller_id, seller_zip_code_prefix, seller_city, seller_state)
category_translation(product_category_name, product_category_name_english)

Join keys:
- orders.customer_id = customers.customer_id
- order_items.order_id = orders.order_id
- order_payments.order_id = orders.order_id
- order_reviews.order_id = orders.order_id
- order_items.product_id = products.product_id
- order_items.seller_id = sellers.seller_id
- products.product_category_name = category_translation.product_category_name

Notes:
- One order can have multiple order_items rows (multiple products per order).
- Delivery delay = order_delivered_customer_date - order_estimated_delivery_date (positive means late).
- product_category_name is in Portuguese; use category_translation for English names.
- Database is SQLite.
"""


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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema_objects = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [row[1] for row in cursor.fetchall()]
        schema_objects[table] = columns
    conn.close()
    return schema_objects


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

    prompt = f"""You are a SQL expert. Given the database schema below, write a single valid SQLite query to answer the question.

{SCHEMA}

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

    schema_objects = get_actual_schema_objects()
    sql_lower = sql.lower()
    referenced_tables = re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_]*)", sql_lower)

    valid_tables = set(t.lower() for t in schema_objects.keys())
    validation_ok = True
    validation_message = "OK"

    if not referenced_tables:
        validation_ok = False
        validation_message = "No table references found in query."
    else:
        for t in referenced_tables:
            if t not in valid_tables:
                validation_ok = False
                validation_message = f"Unknown table referenced: '{t}'"
                break

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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(state["sql"])
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
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
    finally:
        conn.close()
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

    # Simple heuristic for chart type based on result shape
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    os.makedirs("charts", exist_ok=True)
    chart_path = os.path.join("charts", "latest_chart.html")

    try:
        if len(df) == 1:
            # Single row -- a bar chart of its numeric values is more useful than a "chart"
            if num_cols:
                fig = px.bar(x=num_cols, y=[df.iloc[0][c] for c in num_cols],
                             title="Result", labels={"x": "Metric", "y": "Value"})
                chart_type = "single_row_bar"
            else:
                state["chart_path"] = None
                state["chart_type"] = None
                print("[CHART AGENT] Single non-numeric row -- no chart generated.")
                return state
        elif cat_cols and num_cols:
            fig = px.bar(df, x=cat_cols[0], y=num_cols[0],
                         title=f"{num_cols[0]} by {cat_cols[0]}")
            chart_type = "bar"
        elif len(num_cols) >= 2:
            fig = px.scatter(df, x=num_cols[0], y=num_cols[1], title="Result")
            chart_type = "scatter"
        else:
            state["chart_path"] = None
            state["chart_type"] = None
            print("[CHART AGENT] Result shape not chartable -- skipped.")
            return state

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

