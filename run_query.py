import sqlite3
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

conn = sqlite3.connect("olist.db")

query = """
SELECT 
    payment_type,
    COUNT(*) AS usage_count
FROM order_payments
GROUP BY payment_type
ORDER BY usage_count DESC
LIMIT 1;
"""

df = pd.read_sql_query(query, conn)
print(df)

conn.close()