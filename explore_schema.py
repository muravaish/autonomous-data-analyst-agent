import sqlite3

conn = sqlite3.connect("olist.db")
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]

for table in tables:
    print(f"\n{'='*60}")
    print(f"TABLE: {table}")
    print('='*60)

    # Column info
    cursor.execute(f"PRAGMA table_info({table});")
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[1]:<30} {col[2]}")

    # Sample row
    cursor.execute(f"SELECT * FROM {table} LIMIT 1;")
    sample = cursor.fetchone()
    print(f"\n  Sample row: {sample}")

conn.close()