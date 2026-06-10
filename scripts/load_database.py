import duckdb
import os

# Connect to (or create) the database file
con = duckdb.connect("data/health.db")

print("Loading tables into DuckDB...")

# Load each CSV as a permanent table
tables = ["patients", "encounters", "conditions", "observations", "medications"]

for table in tables:
    csv_path = f"data/{table}.csv"
    con.execute(f"""
        CREATE OR REPLACE TABLE {table}
        AS SELECT * FROM read_csv_auto('{csv_path}')
    """)
    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  Loaded {table}: {count} rows")

print("\nAll tables loaded into data/health.db")
print("You can now query them with SQL!")

con.close()