import sqlite3
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

conn = sqlite3.connect("olist.db")

queries = {
    "gold_003": """
        SELECT COUNT(*) AS late_orders
        FROM orders
        WHERE order_delivered_customer_date IS NOT NULL
        AND order_delivered_customer_date > order_estimated_delivery_date;
    """,
    "gold_004": """
        SELECT AVG(r.review_score) AS avg_score_when_late
        FROM order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        WHERE o.order_delivered_customer_date > o.order_estimated_delivery_date;
    """,
    "gold_005": """
        SELECT ct.product_category_name_english AS category,
       AVG(julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date)) AS avg_delay_days
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
JOIN category_translation ct ON p.product_category_name = ct.product_category_name
WHERE o.order_status = 'delivered' 
  AND o.order_delivered_customer_date IS NOT NULL
  AND o.order_delivered_customer_date > o.order_estimated_delivery_date
GROUP BY ct.product_category_name_english
ORDER BY avg_delay_days DESC
LIMIT 1;
    """,
    "gold_006": """
        SELECT COUNT(DISTINCT customer_unique_id) AS unique_customers
        FROM customers;
    """,
    "gold_007": """
        SELECT customer_state, COUNT(*) AS cnt
        FROM customers
        GROUP BY customer_state
        ORDER BY cnt DESC
        LIMIT 1;
    """,
    "gold_008": """
        SELECT AVG(freight_value / price) * 100 AS avg_freight_pct
        FROM order_items
        WHERE price > 0;
    """,
    "gold_009": """
        SELECT seller_id,
               SUM(price + freight_value) AS total_revenue,
               COUNT(DISTINCT order_id) AS distinct_orders
        FROM order_items
        GROUP BY seller_id
        ORDER BY total_revenue DESC
        LIMIT 1;
    """,
    "gold_010": """
        SELECT SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_late
        FROM order_reviews r
        JOIN orders o ON r.order_id = o.order_id
        WHERE r.review_score = 1 AND o.order_delivered_customer_date IS NOT NULL;
    """,
}

for qid, sql in queries.items():
    print(f"\n=== {qid} ===")
    df = pd.read_sql_query(sql, conn)
    print(df)

conn.close()