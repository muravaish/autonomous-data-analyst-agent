import os
import re
import time

from dotenv import load_dotenv
from google import genai

from db_adapter import execute_query, get_database_dialect, get_schema_objects, get_schema_text, validate_sql

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_actual_schema_objects():
    return get_schema_objects()

OLIST_SQL_GUIDANCE = """
Business metric definitions for Olist e-commerce questions:
- total_revenue / seller revenue / product revenue = SUM(order_items.price + order_items.freight_value), not price alone.
- item_revenue = SUM(order_items.price) only when the question explicitly excludes freight.
- distinct_orders = COUNT(DISTINCT order_items.order_id).
- late delivery means orders.order_delivered_customer_date > orders.order_estimated_delivery_date.
- For late-delivery rate or percentage, exclude rows where order_delivered_customer_date is NULL.
- For delivered-order delivery-delay questions, filter orders.order_status = 'delivered' when the orders table is available.
- delivery_delay_days = julianday(orders.order_delivered_customer_date) - julianday(orders.order_estimated_delivery_date) in SQLite.
- average_review_score = AVG(order_reviews.review_score).
- cancellation_rate = canceled orders divided by total relevant orders.
- product category names should use category_translation.product_category_name_english when that table is available.
- Join products to category_translation with products.product_category_name = category_translation.product_category_name.
- Count review rows with COUNT(*) unless the question explicitly asks for distinct orders.
"""


def schema_has_olist_tables(schema_text: str) -> bool:
    lowered = schema_text.lower()
    return "orders" in lowered and "order_items" in lowered


def sql_business_guidance(schema_text: str) -> str:
    if schema_has_olist_tables(schema_text):
        return OLIST_SQL_GUIDANCE
    return "For uploaded datasets, infer metric definitions only from the provided columns and avoid assuming Olist-specific tables."


def call_gemini_with_retry(prompt: str, max_retries: int = 5) -> str:
    """Calls Gemini with retries and falls back to flash-lite during congestion."""
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


def generate_sql(question: str) -> str:
    dialect = get_database_dialect()
    schema_text = get_schema_text()

    business_guidance = sql_business_guidance(schema_text)

    prompt = f"""You are a SQL expert. Given the database schema below, write a single valid {dialect} query to answer the question.

{schema_text}

{business_guidance}

Question: {question}

Rules:
- Return ONLY the SQL query, no explanation, no markdown formatting, no ```sql fences.
- Use only tables and columns listed above.
- Use clean, business-friendly column aliases for calculated fields, such as avg_review_score or total_revenue; do not expose raw expressions like AVG(t1.review_score) as result column names.
- End the query with a semicolon.
"""
    sql = call_gemini_with_retry(prompt)
    return re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.MULTILINE).strip()


def validate_generated_sql(sql: str, schema_objects: dict | None = None) -> tuple[bool, str]:
    return validate_sql(sql)


def execute_sql(sql: str):
    return execute_query(sql)


def run_pipeline(question: str):
    print(f"\nQUESTION: {question}")

    sql = generate_sql(question)
    print(f"\nGENERATED SQL:\n{sql}")

    schema_objects = get_actual_schema_objects()
    valid, message = validate_generated_sql(sql, schema_objects)
    print(f"\nVALIDATION: {message}")

    if not valid:
        print("Stopping -- SQL failed validation.")
        return

    columns, rows = execute_sql(sql)
    print(f"\nRESULT COLUMNS: {columns}")
    print(f"RESULT ROWS: {rows}")


if __name__ == "__main__":
    run_pipeline("Which payment method is used most frequently by customers, by number of transactions?")

