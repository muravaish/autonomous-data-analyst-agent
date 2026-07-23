import sqlite3
import pandas as pd
import os

DATA_DIR = "data"
DB_PATH = "olist.db"

# Map CSV files to table names
files_to_tables = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "category_translation",
}

conn = sqlite3.connect(DB_PATH)

for filename, table_name in files_to_tables.items():
    filepath = os.path.join(DATA_DIR, filename)
    print(f"Loading {filename} -> table '{table_name}'...")
    df = pd.read_csv(filepath)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"  -> {len(df)} rows loaded")

conn.close()
print("\nDone. Database created at:", DB_PATH)