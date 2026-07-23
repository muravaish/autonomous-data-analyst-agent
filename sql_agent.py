import os
import sqlite3
import re
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

DB_PATH = "olist.db"

# --- Schema description the AI will see ---
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


def get_actual_schema_objects():
    """Get real table and column names from the DB, for validation."""
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


def call_gemini_with_retry(prompt: str, max_retries: int = 5) -> str:
    """Calls Gemini with retries + increasing wait time + a fallback model.

    Tries the standard flash model twice, then falls back to the lighter
    flash-lite model for the remaining attempts (it tends to have more
    headroom during peak-demand 503 windows on the free tier).
    """
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
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)  # 10s, 20s, 30s, 40s...
                print(f"  ({model} busy, retrying in {wait}s... attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
    # If we exhausted retries, raise the last error
    raise last_error


def generate_sql(question: str) -> str:
    prompt = f"""You are a SQL expert. Given the database schema below, write a single valid SQLite query to answer the question.

{SCHEMA}

Question: {question}

Rules:
- Return ONLY the SQL query, no explanation, no markdown formatting, no ```sql fences.
- Use only tables and columns listed above.
- Use clean, business-friendly column aliases for calculated fields, such as avg_review_score or total_revenue; do not expose raw expressions like AVG(t1.review_score) as result column names.
- End the query with a semicolon.
"""
    sql = call_gemini_with_retry(prompt)
    # Strip markdown fences if the model added them anyway
    sql = re.sub(r"^```sql\s*|^```\s*|```$", "", sql, flags=re.MULTILINE).strip()
    return sql


def validate_sql(sql: str, schema_objects: dict) -> tuple[bool, str]:
    """Schema-aware check: does every table mentioned actually exist?
    (Basic version -- checks tables referenced via FROM/JOIN)"""
    sql_lower = sql.lower()

    referenced_tables = re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_]*)", sql_lower)
    if not referenced_tables:
        return False, "No table references found in query."

    valid_tables = set(t.lower() for t in schema_objects.keys())
    for t in referenced_tables:
        if t not in valid_tables:
            return False, f"Unknown table referenced: '{t}'"

    return True, "OK"


def execute_sql(sql: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return columns, rows
    finally:
        conn.close()


def run_pipeline(question: str):
    print(f"\nQUESTION: {question}")

    sql = generate_sql(question)
    print(f"\nGENERATED SQL:\n{sql}")

    schema_objects = get_actual_schema_objects()
    valid, message = validate_sql(sql, schema_objects)
    print(f"\nVALIDATION: {message}")

    if not valid:
        print("Stopping -- SQL failed validation.")
        return

    columns, rows = execute_sql(sql)
    print(f"\nRESULT COLUMNS: {columns}")
    print(f"RESULT ROWS: {rows}")


if __name__ == "__main__":
    run_pipeline("Which payment method is used most frequently by customers, by number of transactions?")
