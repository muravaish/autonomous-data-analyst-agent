from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

READ_ONLY_STARTERS = ("select", "with")
BLOCKED_SQL_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "attach",
    "detach",
    "vacuum",
    "pragma",
    "replace",
    "merge",
    "grant",
    "revoke",
}

OLIST_JOIN_NOTES = """
Join keys:
- orders.customer_id = customers.customer_id
- order_items.order_id = orders.order_id
- order_payments.order_id = orders.order_id
- order_reviews.order_id = orders.order_id
- order_items.product_id = products.product_id
- order_items.seller_id = sellers.seller_id
- products.product_category_name = category_translation.product_category_name

Business notes:
- One order can have multiple order_items rows, so revenue questions often need order_items.
- Delivery delay means delivered after estimated delivery date.
- product_category_name is in Portuguese; use category_translation for English category names when available.
""".strip()


def get_database_backend() -> str:
    return os.getenv("DATABASE_BACKEND", "sqlite").strip().lower()


def get_database_dialect() -> str:
    backend = get_database_backend()
    if backend in {"postgres", "postgresql", "azure_postgres", "azure-postgres"}:
        return "PostgreSQL"
    return "SQLite"


def get_sqlite_path() -> str:
    return os.getenv("SQLITE_DB_PATH", "olist.db")


@contextmanager
def get_connection() -> Iterator[Any]:
    backend = get_database_backend()
    if backend == "sqlite":
        conn = sqlite3.connect(get_sqlite_path())
        try:
            yield conn
        finally:
            conn.close()
        return

    if backend in {"postgres", "postgresql", "azure_postgres", "azure-postgres"}:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is required when DATABASE_BACKEND is postgres.")
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError("Install psycopg to use Azure PostgreSQL: pip install 'psycopg[binary]'.") from exc

        conn = psycopg.connect(database_url)
        try:
            yield conn
        finally:
            conn.close()
        return

    raise ValueError(f"Unsupported DATABASE_BACKEND: {backend}")


def _quote_identifier(identifier: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe identifier: {identifier}")
    if get_database_dialect() == "PostgreSQL":
        return f'"{identifier}"'
    return f'"{identifier}"'


def get_schema_objects() -> dict[str, list[str]]:
    backend = get_database_backend()
    schema_objects: dict[str, list[str]] = {}

    with get_connection() as conn:
        cursor = conn.cursor()
        if backend == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"PRAGMA table_info({_quote_identifier(table)});")
                schema_objects[table] = [row[1] for row in cursor.fetchall()]
            return schema_objects

        database_schema = os.getenv("DATABASE_SCHEMA", "public")
        cursor.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position;
            """,
            (database_schema,),
        )
        for table_name, column_name in cursor.fetchall():
            schema_objects.setdefault(table_name, []).append(column_name)

    return schema_objects


def get_schema_text() -> str:
    schema_objects = get_schema_objects()
    table_lines = ["Tables and columns:"]
    for table, columns in sorted(schema_objects.items()):
        table_lines.append(f"{table}({', '.join(columns)})")

    notes = [f"SQL dialect: {get_database_dialect()}."]
    olist_tables = {"customers", "orders", "order_items", "order_payments", "order_reviews", "products", "sellers", "category_translation"}
    if olist_tables.intersection(schema_objects.keys()):
        notes.append(OLIST_JOIN_NOTES)

    return "\n".join(table_lines + ["", *notes])


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()


def validate_sql(sql: str) -> tuple[bool, str]:
    cleaned = _strip_sql_comments(sql)
    if not cleaned:
        return False, "Empty SQL query."

    lowered = cleaned.lower().strip()
    if not lowered.startswith(READ_ONLY_STARTERS):
        return False, "Only read-only SELECT/WITH queries are allowed."

    keyword_hits = re.findall(r"\b[a-z_]+\b", lowered)
    blocked = sorted(BLOCKED_SQL_KEYWORDS.intersection(keyword_hits))
    if blocked:
        return False, f"Blocked non-read-only SQL keyword(s): {', '.join(blocked)}"

    schema_objects = get_schema_objects()
    referenced_tables = re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_]*)", lowered)
    if not referenced_tables:
        return False, "No table references found in query."

    valid_tables = {table.lower() for table in schema_objects}
    for table in referenced_tables:
        if table not in valid_tables:
            return False, f"Unknown table referenced: '{table}'"

    return True, "OK"


def execute_query(sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    validation_ok, validation_message = validate_sql(sql)
    if not validation_ok:
        raise ValueError(f"SQL validation failed: {validation_message}")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        description = cursor.description or []
        columns = [desc[0] for desc in description]
        rows = cursor.fetchall()
    return columns, rows
