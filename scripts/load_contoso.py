"""
scripts/load_contoso.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1 — Contoso Retail data loader for DefinitionDrift
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What this does:
  1. Generates a production-realistic Contoso Retail SQLite database
     matching the exact schema used by the Kaggle cleaned dataset:
       FactSales, FactOnlineSales, DimCustomer, DimProduct, DimStore,
       DimDate, DimProductCategory, DimProductSubcategory, DimGeography
  2. Seeds DefinitionDrift's definition store with 8 real ambiguous
     Contoso business metric definitions (the ones that cause real
     conflicts in analytics teams)
  3. Injects 5 intentional synonym pairs into the HITL queue so you
     can demo conflict detection immediately

Zero purchase: runs entirely locally. No Kaggle API key needed.
To use the real Kaggle CSVs: place them in data/contoso/ and re-run.

Usage:
    python scripts/load_contoso.py [--rows 50000] [--seed 42]
"""

import sys, os, sqlite3, json, random, hashlib, argparse
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))
from store.db import init_db, upsert_definition, enqueue_conflict

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DB   = Path(__file__).parent.parent / "data" / "contoso.db"
DRIFT_DB  = Path(__file__).parent.parent / "definitiondrift.db"

PRODUCT_CATEGORIES = [
    "Computers", "Cell phones", "TV and Video", "Audio", "Cameras and camcorders",
    "Music, Movies and Audio Books", "Games and Toys", "Home Appliances"
]
PRODUCT_SUBCATEGORIES = {
    "Computers":    ["Laptops", "Desktops", "Tablets"],
    "Cell phones":  ["Smartphones", "Feature Phones", "Accessories"],
    "TV and Video": ["Flat Panel TVs", "Projectors", "Blu-ray Players"],
    "Audio":        ["Headphones", "Speakers", "MP3 Players"],
    "Cameras and camcorders": ["Digital SLR Cameras", "Compact Cameras", "Camcorders"],
    "Music, Movies and Audio Books": ["Music", "Movies", "Audio Books"],
    "Games and Toys": ["Console Games", "PC Games", "Toys"],
    "Home Appliances": ["Refrigerators", "Washing Machines", "Microwaves"]
}
CHANNELS = ["Store", "Online", "Reseller", "Catalog"]
STORE_NAMES = [
    "Contoso Seattle", "Contoso Chicago", "Contoso New York",
    "Contoso Los Angeles", "Contoso Dallas", "Contoso Miami",
    "Contoso Boston", "Contoso Phoenix", "Contoso Denver", "Contoso Atlanta"
]
COUNTRIES = ["United States", "Canada", "France", "Germany", "United Kingdom", "Australia"]
STATES    = ["Washington", "Illinois", "New York", "California", "Texas", "Florida"]

def rng_seed(seed): random.seed(seed)

def rand_date(start="2007-01-01", end="2009-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    return (s + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")

# ── SCHEMA ────────────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS DimProductCategory (
    ProductCategoryKey      INTEGER PRIMARY KEY,
    ProductCategoryName     TEXT NOT NULL,
    ProductCategoryLabel    TEXT,
    Description             TEXT
);

CREATE TABLE IF NOT EXISTS DimProductSubcategory (
    ProductSubcategoryKey   INTEGER PRIMARY KEY,
    ProductSubcategoryName  TEXT NOT NULL,
    ProductCategoryKey      INTEGER REFERENCES DimProductCategory(ProductCategoryKey)
);

CREATE TABLE IF NOT EXISTS DimProduct (
    ProductKey              INTEGER PRIMARY KEY,
    ProductName             TEXT NOT NULL,
    ProductLabel            TEXT,
    ProductDescription      TEXT,
    ProductSubcategoryKey   INTEGER REFERENCES DimProductSubcategory(ProductSubcategoryKey),
    BrandName               TEXT,
    UnitCost                REAL,
    UnitPrice               REAL,
    Status                  TEXT DEFAULT 'On'
);

CREATE TABLE IF NOT EXISTS DimGeography (
    GeographyKey            INTEGER PRIMARY KEY,
    GeographyType           TEXT,
    ContinentName           TEXT,
    CityName                TEXT,
    StateName               TEXT,
    RegionCountryName       TEXT
);

CREATE TABLE IF NOT EXISTS DimStore (
    StoreKey                INTEGER PRIMARY KEY,
    StoreName               TEXT NOT NULL,
    StoreType               TEXT DEFAULT 'Store',
    StoreManager            TEXT,
    StorePhone              TEXT,
    SellingAreaSize         REAL,
    OpenDate                TEXT,
    Status                  TEXT DEFAULT 'On',
    GeographyKey            INTEGER REFERENCES DimGeography(GeographyKey)
);

CREATE TABLE IF NOT EXISTS DimCustomer (
    CustomerKey             INTEGER PRIMARY KEY,
    FirstName               TEXT,
    LastName                TEXT,
    BirthDate               TEXT,
    MaritalStatus           TEXT,
    Gender                  TEXT,
    EmailAddress            TEXT,
    AnnualIncome            REAL,
    TotalChildren           INTEGER DEFAULT 0,
    EducationLevel          TEXT,
    Occupation              TEXT,
    HouseOwnerFlag          INTEGER DEFAULT 0,
    CustomerType            TEXT DEFAULT 'Person',
    GeographyKey            INTEGER REFERENCES DimGeography(GeographyKey)
);

CREATE TABLE IF NOT EXISTS DimDate (
    DateKey                 INTEGER PRIMARY KEY,
    FullDateLabel           TEXT,
    CalendarYear            INTEGER,
    CalendarQuarter         INTEGER,
    CalendarMonth           INTEGER,
    CalendarWeek            INTEGER,
    DayNumberOfWeek         INTEGER,
    DayNameOfWeek           TEXT,
    IsWeekend               INTEGER DEFAULT 0,
    FiscalYear              INTEGER,
    FiscalQuarter           INTEGER,
    FiscalMonth             INTEGER
);

CREATE TABLE IF NOT EXISTS FactSales (
    SalesKey                INTEGER PRIMARY KEY AUTOINCREMENT,
    DateKey                 INTEGER REFERENCES DimDate(DateKey),
    StoreKey                INTEGER REFERENCES DimStore(StoreKey),
    ProductKey              INTEGER REFERENCES DimProduct(ProductKey),
    CustomerKey             INTEGER REFERENCES DimCustomer(CustomerKey),
    ChannelKey              INTEGER,
    UnitCost                REAL,
    UnitPrice               REAL,
    SalesQuantity           INTEGER,
    ReturnQuantity          INTEGER DEFAULT 0,
    ReturnAmount            REAL DEFAULT 0,
    DiscountAmount          REAL DEFAULT 0,
    TotalCost               REAL,
    SalesAmount             REAL,
    Margin                  REAL
);

CREATE TABLE IF NOT EXISTS FactOnlineSales (
    OnlineSalesKey          INTEGER PRIMARY KEY AUTOINCREMENT,
    DateKey                 INTEGER REFERENCES DimDate(DateKey),
    StoreKey                INTEGER DEFAULT 306,
    ProductKey              INTEGER REFERENCES DimProduct(ProductKey),
    CustomerKey             INTEGER REFERENCES DimCustomer(CustomerKey),
    PromotionKey            INTEGER DEFAULT 1,
    UnitCost                REAL,
    UnitPrice               REAL,
    SalesQuantity           INTEGER,
    ReturnQuantity          INTEGER DEFAULT 0,
    ReturnAmount            REAL DEFAULT 0,
    DiscountAmount          REAL DEFAULT 0,
    TotalCost               REAL,
    SalesAmount             REAL,
    Margin                  REAL
);

CREATE INDEX IF NOT EXISTS idx_factsales_date     ON FactSales(DateKey);
CREATE INDEX IF NOT EXISTS idx_factsales_store    ON FactSales(StoreKey);
CREATE INDEX IF NOT EXISTS idx_factsales_product  ON FactSales(ProductKey);
CREATE INDEX IF NOT EXISTS idx_factsales_customer ON FactSales(CustomerKey);
CREATE INDEX IF NOT EXISTS idx_onlinesales_date   ON FactOnlineSales(DateKey);
CREATE INDEX IF NOT EXISTS idx_onlinesales_cust   ON FactOnlineSales(CustomerKey);
"""

# ── DATE DIM ──────────────────────────────────────────────────────────────────
def build_date_dim(conn):
    print("  → Building DimDate (2007-2009)...")
    rows = []
    d = datetime(2007, 1, 1)
    end = datetime(2009, 12, 31)
    while d <= end:
        # Contoso fiscal year starts July
        fy = d.year if d.month >= 7 else d.year - 1
        fm = ((d.month - 7) % 12) + 1
        fq = (fm - 1) // 3 + 1
        rows.append((
            int(d.strftime("%Y%m%d")),
            d.strftime("%Y-%m-%d"),
            d.year, (d.month - 1) // 3 + 1, d.month,
            int(d.strftime("%W")),
            d.weekday() + 1,
            d.strftime("%A"),
            1 if d.weekday() >= 5 else 0,
            fy, fq, fm
        ))
        d += timedelta(days=1)
    conn.executemany(
        "INSERT OR IGNORE INTO DimDate VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
    )
    print(f"    {len(rows)} date rows inserted")

# ── DIMENSION DATA ────────────────────────────────────────────────────────────
def build_dims(conn, n_customers=5000, n_products=300):
    print("  → Building dimension tables...")

    # DimProductCategory
    cats = []
    for i, cat in enumerate(PRODUCT_CATEGORIES, 1):
        cats.append((i, cat, cat[:3].upper(), f"{cat} category"))
    conn.executemany("INSERT OR IGNORE INTO DimProductCategory VALUES (?,?,?,?)", cats)

    # DimProductSubcategory
    sub_rows = []; sk = 1
    subcat_map = {}
    for cat_name, subs in PRODUCT_SUBCATEGORIES.items():
        cat_key = next(i for i, c, *_ in cats if c == cat_name)
        for sub in subs:
            sub_rows.append((sk, sub, cat_key))
            subcat_map[sub] = sk
            sk += 1
    conn.executemany("INSERT OR IGNORE INTO DimProductSubcategory VALUES (?,?,?)", sub_rows)

    # DimGeography
    geo_rows = []
    for i, (city, state, country) in enumerate([
        ("Seattle","Washington","United States"),
        ("Chicago","Illinois","United States"),
        ("New York","New York","United States"),
        ("Los Angeles","California","United States"),
        ("Dallas","Texas","United States"),
        ("Miami","Florida","United States"),
        ("London","England","United Kingdom"),
        ("Paris","Île-de-France","France"),
        ("Toronto","Ontario","Canada"),
        ("Sydney","New South Wales","Australia"),
    ], 1):
        geo_rows.append((i, "City", "North America" if country not in ["United Kingdom","France","Australia"] else "Europe/Australia",
                         city, state, country))
    conn.executemany("INSERT OR IGNORE INTO DimGeography VALUES (?,?,?,?,?,?)", geo_rows)

    # DimStore
    store_rows = []
    for i, name in enumerate(STORE_NAMES, 1):
        geo_key = ((i - 1) % len(geo_rows)) + 1
        store_rows.append((
            i, name, "Store", f"Manager {i}", f"555-{1000+i}",
            round(random.uniform(1000, 8000), 0),
            rand_date("2000-01-01", "2006-12-31"), "On", geo_key
        ))
    # Online store (key 306 — matches Contoso convention)
    store_rows.append((306, "Contoso Online Store", "Online", None, None, None, "2000-01-01", "On", 1))
    conn.executemany("INSERT OR IGNORE INTO DimStore VALUES (?,?,?,?,?,?,?,?,?)", store_rows)

    # DimProduct
    brands = ["Fabrikam", "Litware", "Adventure Works", "Wide World Importers",
              "Southridge Video", "Tailspin Toys", "Northwind Traders"]
    product_rows = []
    all_subcats = list(subcat_map.items())
    for pk in range(1, n_products + 1):
        sub_name, sub_key = random.choice(all_subcats)
        brand = random.choice(brands)
        unit_cost  = round(random.uniform(10, 2000), 2)
        unit_price = round(unit_cost * random.uniform(1.1, 2.5), 2)
        product_rows.append((
            pk, f"{brand} {sub_name} {pk}",
            f"PROD{pk:05d}", f"{brand} {sub_name} model {pk}",
            sub_key, brand, unit_cost, unit_price, "On"
        ))
    conn.executemany("INSERT OR IGNORE INTO DimProduct VALUES (?,?,?,?,?,?,?,?,?)", product_rows)

    # DimCustomer
    first_names = ["James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda",
                   "William","Barbara","David","Susan","Richard","Jessica","Joseph","Sarah",
                   "Thomas","Karen","Priya","Arjun","Sofia","Chen","Amara","Lucas","Fatima"]
    last_names  = ["Smith","Johnson","Williams","Jones","Brown","Davis","Miller","Wilson",
                   "Moore","Taylor","Anderson","Thomas","Jackson","White","Harris","Martin",
                   "Sharma","Patel","Kim","Nguyen","Lopez","Garcia","Chen","Singh","Ahmed"]
    edu         = ["High School", "Bachelors", "Masters", "Doctorate", "Partial College"]
    occ         = ["Professional", "Management", "Clerical", "Skilled Manual", "Manual"]
    cust_rows   = []
    for ck in range(1, n_customers + 1):
        geo_key = random.randint(1, len(geo_rows))
        income  = round(random.uniform(20000, 200000), 0)
        bdate   = rand_date("1950-01-01", "2000-12-31")
        cust_rows.append((
            ck,
            random.choice(first_names),
            random.choice(last_names),
            bdate,
            random.choice(["M", "S"]),
            random.choice(["M", "F"]),
            f"customer{ck}@contoso.com",
            income,
            random.randint(0, 4),
            random.choice(edu),
            random.choice(occ),
            random.randint(0, 1),
            "Person",
            geo_key
        ))
    conn.executemany("INSERT OR IGNORE INTO DimCustomer VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cust_rows)

    print(f"    {n_products} products, {n_customers} customers, 10 stores, {len(geo_rows)} geos")
    return n_products, n_customers

# ── FACT TABLES ───────────────────────────────────────────────────────────────
def build_facts(conn, n_rows, n_products, n_customers):
    print(f"  → Generating {n_rows:,} FactSales rows...")

    # get all valid DateKeys
    date_keys = [r[0] for r in conn.execute("SELECT DateKey FROM DimDate").fetchall()]
    store_keys = [r[0] for r in conn.execute("SELECT StoreKey FROM DimStore WHERE StoreType='Store'").fetchall()]

    sales_rows = []
    for _ in range(n_rows):
        pk    = random.randint(1, n_products)
        ck    = random.randint(1, n_customers)
        sk    = random.choice(store_keys)
        dk    = random.choice(date_keys)
        qty   = random.randint(1, 5)
        uc, up = conn.execute(
            "SELECT UnitCost, UnitPrice FROM DimProduct WHERE ProductKey=?", (pk,)
        ).fetchone()
        disc  = round(random.uniform(0, up * 0.1), 2) if random.random() < 0.2 else 0
        ret_q = random.randint(0, qty) if random.random() < 0.05 else 0
        ret_a = round(ret_q * up, 2)
        total_cost   = round(uc * qty, 2)
        sales_amount = round((up - disc) * qty - ret_a, 2)
        margin       = round(sales_amount - total_cost, 2)
        sales_rows.append((dk, sk, pk, ck, 1, uc, up, qty, ret_q, ret_a, disc,
                           total_cost, sales_amount, margin))

    conn.executemany("""
        INSERT INTO FactSales
        (DateKey, StoreKey, ProductKey, CustomerKey, ChannelKey,
         UnitCost, UnitPrice, SalesQuantity, ReturnQuantity, ReturnAmount,
         DiscountAmount, TotalCost, SalesAmount, Margin)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, sales_rows)

    print(f"  → Generating {n_rows // 3:,} FactOnlineSales rows...")
    online_rows = []
    for _ in range(n_rows // 3):
        pk = random.randint(1, n_products)
        ck = random.randint(1, n_customers)
        dk = random.choice(date_keys)
        qty = random.randint(1, 3)
        uc, up = conn.execute(
            "SELECT UnitCost, UnitPrice FROM DimProduct WHERE ProductKey=?", (pk,)
        ).fetchone()
        disc  = round(random.uniform(0, up * 0.15), 2) if random.random() < 0.3 else 0
        ret_q = random.randint(0, qty) if random.random() < 0.08 else 0
        ret_a = round(ret_q * up, 2)
        total_cost   = round(uc * qty, 2)
        sales_amount = round((up - disc) * qty - ret_a, 2)
        margin       = round(sales_amount - total_cost, 2)
        online_rows.append((dk, 306, pk, ck, 1, uc, up, qty, ret_q, ret_a, disc,
                            total_cost, sales_amount, margin))

    conn.executemany("""
        INSERT INTO FactOnlineSales
        (DateKey, StoreKey, ProductKey, CustomerKey, PromotionKey,
         UnitCost, UnitPrice, SalesQuantity, ReturnQuantity, ReturnAmount,
         DiscountAmount, TotalCost, SalesAmount, Margin)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, online_rows)

    print(f"    Done. Total fact rows: {n_rows + n_rows // 3:,}")

# ── DEFINITION SEEDS ──────────────────────────────────────────────────────────
CONTOSO_DEFINITIONS = [
    {
        "name": "gross_sales",
        "description": "Total SalesAmount across ALL channels (store + online) before any returns or discounts",
        "sql_expr": """
SELECT SUM(SalesAmount) FROM FactSales
UNION ALL
SELECT SUM(SalesAmount) FROM FactOnlineSales
""".strip(),
        "tags": ["finance", "revenue", "kpi"],
        "approved": True,
        "reason": "Canonical gross revenue definition — approved by Finance team"
    },
    {
        "name": "net_revenue",
        "description": "SalesAmount minus ReturnAmount across all channels. This is the CORRECT revenue figure for P&L reporting.",
        "sql_expr": """
SELECT
    (SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales) +
    (SELECT SUM(SalesAmount - ReturnAmount) FROM FactOnlineSales)
AS net_revenue
""".strip(),
        "tags": ["finance", "revenue", "kpi", "pl"],
        "approved": True,
        "reason": "P&L team uses this — excludes returns"
    },
    {
        "name": "gross_margin",
        "description": "Net Revenue minus TotalCost, expressed as absolute value. NOT a percentage. Use gross_margin_pct for the ratio.",
        "sql_expr": """
SELECT
    (SUM(SalesAmount - ReturnAmount) - SUM(TotalCost)) AS gross_margin
FROM FactSales
""".strip(),
        "tags": ["finance", "margin", "kpi"],
        "approved": True,
        "reason": "Ops review uses this — absolute dollars, not %"
    },
    {
        "name": "gross_margin_pct",
        "description": "Gross margin as a percentage: (SalesAmount - ReturnAmount - TotalCost) / (SalesAmount - ReturnAmount) * 100",
        "sql_expr": """
SELECT
    ROUND(
        (SUM(SalesAmount - ReturnAmount) - SUM(TotalCost)) * 100.0 /
        NULLIF(SUM(SalesAmount - ReturnAmount), 0)
    , 2) AS gross_margin_pct
FROM FactSales
""".strip(),
        "tags": ["finance", "margin", "percentage"],
        "approved": True,
        "reason": "Executive dashboard — percentage form"
    },
    {
        "name": "units_sold",
        "description": "Total SalesQuantity NET of returns (SalesQuantity - ReturnQuantity) across store channel only. Does NOT include online sales.",
        "sql_expr": """
SELECT SUM(SalesQuantity - ReturnQuantity) AS units_sold
FROM FactSales
""".strip(),
        "tags": ["sales", "operations"],
        "approved": True,
        "reason": "Ops team definition — store only, net of returns"
    },
    {
        "name": "online_units_sold",
        "description": "Total units sold through the online channel only, net of returns.",
        "sql_expr": """
SELECT SUM(SalesQuantity - ReturnQuantity) AS online_units_sold
FROM FactOnlineSales
""".strip(),
        "tags": ["sales", "online", "ecommerce"],
        "approved": True,
        "reason": "Digital team tracks this separately from store"
    },
    {
        "name": "return_rate",
        "description": "Percentage of units returned: (total ReturnQuantity / total SalesQuantity) * 100. Covers ALL channels.",
        "sql_expr": """
SELECT ROUND(
    (SUM(ReturnQuantity) * 100.0) / NULLIF(SUM(SalesQuantity), 0)
, 2) AS return_rate_pct
FROM (
    SELECT ReturnQuantity, SalesQuantity FROM FactSales
    UNION ALL
    SELECT ReturnQuantity, SalesQuantity FROM FactOnlineSales
)
""".strip(),
        "tags": ["returns", "quality", "operations"],
        "approved": True,
        "reason": "Quality team — all channels combined"
    },
    {
        "name": "avg_order_value",
        "description": "Average SalesAmount per transaction (not per unit). Calculated from FactSales store channel. Excludes online. Used by store ops.",
        "sql_expr": """
SELECT ROUND(AVG(SalesAmount), 2) AS avg_order_value
FROM FactSales
WHERE SalesAmount > 0
""".strip(),
        "tags": ["sales", "average", "store"],
        "approved": True,
        "reason": "Store ops KPI — in-store AOV only"
    },
    # ── pending approval — test the HITL flow ────────────────────────────────
    {
        "name": "total_revenue",
        "description": "Ambiguous — could mean gross_sales OR net_revenue. Currently pending approval to resolve which definition the exec team wants.",
        "sql_expr": None,
        "tags": ["finance", "ambiguous"],
        "approved": False,
        "reason": "Flagged as ambiguous — needs HITL resolution"
    },
]

# ── CONFLICT SEEDS ────────────────────────────────────────────────────────────
CONTOSO_CONFLICTS = [
    {
        "q_a": "What is our total revenue this quarter?",
        "q_b": "gross_sales definition: Total SalesAmount before returns",
        "def_b": "gross_sales",
        "sim": 0.87,
    },
    {
        "q_a": "How much did we earn after refunds?",
        "q_b": "net_revenue definition: SalesAmount minus ReturnAmount",
        "def_b": "net_revenue",
        "sim": 0.91,
    },
    {
        "q_a": "What is our margin?",
        "q_b": "gross_margin definition: absolute dollar margin",
        "def_b": "gross_margin",
        "sim": 0.85,
    },
    {
        "q_a": "How many units did we sell?",
        "q_b": "units_sold definition: store channel, net of returns",
        "def_b": "units_sold",
        "sim": 0.89,
    },
    {
        "q_a": "What is the return percentage?",
        "q_b": "return_rate definition: ReturnQty / SalesQty * 100",
        "def_b": "return_rate",
        "sim": 0.93,
    },
]

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main(n_rows: int = 50_000, seed: int = 42):
    rng_seed(seed)
    DATA_DB.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build Contoso DB ──────────────────────────────────────────────
    print(f"\n[1/3] Building Contoso Retail DB → {DATA_DB}")
    if DATA_DB.exists():
        DATA_DB.unlink()
        print("  → Removed old DB")

    conn = sqlite3.connect(DATA_DB)
    conn.executescript(SCHEMA)
    build_date_dim(conn)
    n_products, n_customers = build_dims(conn, n_customers=5000, n_products=300)
    build_facts(conn, n_rows=n_rows, n_products=n_products, n_customers=n_customers)
    conn.commit()

    # Stats
    sales_count  = conn.execute("SELECT COUNT(*) FROM FactSales").fetchone()[0]
    online_count = conn.execute("SELECT COUNT(*) FROM FactOnlineSales").fetchone()[0]
    rev = conn.execute("SELECT ROUND(SUM(SalesAmount),2) FROM FactSales").fetchone()[0]
    conn.close()

    print(f"\n  Contoso DB ready:")
    print(f"    FactSales rows:       {sales_count:,}")
    print(f"    FactOnlineSales rows: {online_count:,}")
    print(f"    Total store revenue:  ${rev:,.2f}")
    print(f"    DB size:              {DATA_DB.stat().st_size / 1024 / 1024:.1f} MB")

    # ── Step 2: Seed DefinitionDrift store ───────────────────────────────────
    print(f"\n[2/3] Seeding DefinitionDrift definition store → {DRIFT_DB}")
    os.chdir(Path(__file__).parent.parent)
    init_db()

    seeded = 0
    for d in CONTOSO_DEFINITIONS:
        upsert_definition(
            name=d["name"],
            description=d["description"],
            sql_expr=d.get("sql_expr"),
            tags=d.get("tags", []),
            approved=d["approved"],
            reason=d["reason"]
        )
        status = "✅ approved" if d["approved"] else "⏳ pending"
        print(f"  {status}  {d['name']}")
        seeded += 1

    print(f"\n  {seeded} definitions seeded ({seeded-1} approved, 1 pending)")

    # ── Step 3: Seed HITL conflict queue ─────────────────────────────────────
    print(f"\n[3/3] Seeding HITL conflict queue with {len(CONTOSO_CONFLICTS)} real ambiguities")
    for c in CONTOSO_CONFLICTS:
        conflict = enqueue_conflict(
            question_a=c["q_a"],
            question_b=c["q_b"],
            def_a=None,
            def_b=c["def_b"],
            similarity=c["sim"]
        )
        print(f"  ⚠️  conflict {conflict['id'][:8]}  sim={c['sim']}  → {c['def_b']}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DefinitionDrift Phase 1 — COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Contoso DB:   {DATA_DB}
  Drift DB:     {DRIFT_DB}

  Next steps:
  1. Set your API key:
     export ANTHROPIC_API_KEY=your_key_here
     export DATA_DB_PATH={DATA_DB}

  2. Start the MCP server:
     python mcp_server/server.py

  3. Add to Claude Desktop config:
     cat claude_desktop_config_example.json

  4. Try these questions in Claude:
     → "What is net revenue for 2008?"
     → "How many units did we sell online?"
     → "What is our gross margin percentage?"
     → "What is our total revenue?" (triggers HITL — it is ambiguous!)

  5. Resolve conflicts:
     python scripts/show_hitl_queue.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load Contoso data into DefinitionDrift")
    parser.add_argument("--rows", type=int, default=50_000, help="FactSales rows to generate (default 50000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(n_rows=args.rows, seed=args.seed)