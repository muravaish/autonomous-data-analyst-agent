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

    prompt = f"""You are a SQL expert. Given the database schema below, write a single valid {dialect} query to answer the question.

{schema_text}

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
