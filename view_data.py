import sqlite3
import pandas as pd

conn = sqlite3.connect("data/contoso.db")

# Show available tables
tables = pd.read_sql("""
    SELECT name
    FROM sqlite_master
    WHERE type='table'
""", conn)

print("Tables:")
print(tables)

# Show first 20 sales records
df = pd.read_sql("SELECT * FROM FactSales LIMIT 20", conn)

print("\nFactSales:")
print(df.to_string(index=False))

conn.close()