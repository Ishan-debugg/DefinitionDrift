"""Quick DB inspector script."""
import sqlite3

c = sqlite3.connect("data/contoso.db")
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

for t in tables:
    cols = [r[1] for r in c.execute(f"PRAGMA table_info({t})").fetchall()]
    count = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"\n--- {t} ({count} rows) ---")
    print(f"  Columns: {cols}")

# Quick sample from FactSales
print("\n--- FactSales sample ---")
for row in c.execute("SELECT * FROM FactSales LIMIT 3").fetchall():
    print(f"  {row}")

# Quick sample from DimDate
print("\n--- DimDate sample ---")
for row in c.execute("SELECT * FROM DimDate LIMIT 3").fetchall():
    print(f"  {row}")

# Quick sample from DimProduct
print("\n--- DimProduct sample ---")
for row in c.execute("SELECT * FROM DimProduct LIMIT 2").fetchall():
    print(f"  {row}")

c.close()
