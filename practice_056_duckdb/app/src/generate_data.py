"""
Generate sample data for DuckDB practice exercises.

Creates a realistic e-commerce dataset with:
- products: 50 products across 5 categories
- stores: 10 stores across 5 regions
- sales: ~100K transactions spanning 2022-2024

Output formats: CSV and Parquet (both individual files and year-partitioned).
This script is FULLY IMPLEMENTED — run it to generate data before starting exercises.
"""

import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
SEED = 42
NUM_SALES = 100_000

CATEGORIES = ["Electronics", "Clothing", "Food", "Books", "Home"]

PRODUCTS: list[dict[str, str | float]] = [
    # Electronics
    {"name": "Laptop Pro 15", "category": "Electronics", "base_price": 1299.99},
    {"name": "Wireless Mouse", "category": "Electronics", "base_price": 29.99},
    {"name": "USB-C Hub", "category": "Electronics", "base_price": 49.99},
    {"name": "Mechanical Keyboard", "category": "Electronics", "base_price": 89.99},
    {"name": "Monitor 27in", "category": "Electronics", "base_price": 399.99},
    {"name": "Webcam HD", "category": "Electronics", "base_price": 69.99},
    {"name": "External SSD 1TB", "category": "Electronics", "base_price": 109.99},
    {"name": "Bluetooth Speaker", "category": "Electronics", "base_price": 44.99},
    {"name": "Tablet 10in", "category": "Electronics", "base_price": 349.99},
    {"name": "Noise-Cancel Headphones", "category": "Electronics", "base_price": 249.99},
    # Clothing
    {"name": "Cotton T-Shirt", "category": "Clothing", "base_price": 19.99},
    {"name": "Denim Jeans", "category": "Clothing", "base_price": 49.99},
    {"name": "Running Shoes", "category": "Clothing", "base_price": 79.99},
    {"name": "Winter Jacket", "category": "Clothing", "base_price": 129.99},
    {"name": "Wool Sweater", "category": "Clothing", "base_price": 59.99},
    {"name": "Silk Scarf", "category": "Clothing", "base_price": 34.99},
    {"name": "Leather Belt", "category": "Clothing", "base_price": 24.99},
    {"name": "Sport Socks 3pk", "category": "Clothing", "base_price": 12.99},
    {"name": "Baseball Cap", "category": "Clothing", "base_price": 17.99},
    {"name": "Rain Boots", "category": "Clothing", "base_price": 54.99},
    # Food
    {"name": "Organic Coffee 1lb", "category": "Food", "base_price": 14.99},
    {"name": "Dark Chocolate Bar", "category": "Food", "base_price": 4.99},
    {"name": "Olive Oil 500ml", "category": "Food", "base_price": 11.99},
    {"name": "Protein Bars 12pk", "category": "Food", "base_price": 24.99},
    {"name": "Green Tea Box", "category": "Food", "base_price": 8.99},
    {"name": "Almond Butter", "category": "Food", "base_price": 9.99},
    {"name": "Dried Mango", "category": "Food", "base_price": 6.99},
    {"name": "Granola Mix", "category": "Food", "base_price": 7.99},
    {"name": "Sparkling Water 12pk", "category": "Food", "base_price": 5.99},
    {"name": "Trail Mix 1lb", "category": "Food", "base_price": 10.99},
    # Books
    {"name": "Python Cookbook", "category": "Books", "base_price": 39.99},
    {"name": "Data Science Handbook", "category": "Books", "base_price": 44.99},
    {"name": "SQL Performance Guide", "category": "Books", "base_price": 34.99},
    {"name": "Clean Code", "category": "Books", "base_price": 29.99},
    {"name": "Design Patterns", "category": "Books", "base_price": 49.99},
    {"name": "DDIA", "category": "Books", "base_price": 42.99},
    {"name": "Algorithm Design Manual", "category": "Books", "base_price": 54.99},
    {"name": "Rust Programming", "category": "Books", "base_price": 37.99},
    {"name": "Machine Learning Yearning", "category": "Books", "base_price": 19.99},
    {"name": "The Art of SQL", "category": "Books", "base_price": 32.99},
    # Home
    {"name": "Desk Lamp LED", "category": "Home", "base_price": 34.99},
    {"name": "Throw Blanket", "category": "Home", "base_price": 29.99},
    {"name": "Ceramic Mug Set", "category": "Home", "base_price": 22.99},
    {"name": "Wall Clock", "category": "Home", "base_price": 19.99},
    {"name": "Plant Pot Large", "category": "Home", "base_price": 15.99},
    {"name": "Scented Candle", "category": "Home", "base_price": 12.99},
    {"name": "Kitchen Timer", "category": "Home", "base_price": 9.99},
    {"name": "Bamboo Cutting Board", "category": "Home", "base_price": 18.99},
    {"name": "Glass Vase", "category": "Home", "base_price": 24.99},
    {"name": "Linen Napkins 4pk", "category": "Home", "base_price": 16.99},
]

STORES: list[dict[str, str]] = [
    {"name": "Downtown Flagship", "city": "New York", "region": "East"},
    {"name": "Mall of America", "city": "Minneapolis", "region": "Central"},
    {"name": "Lakeside Plaza", "city": "Chicago", "region": "Central"},
    {"name": "Sunset Strip", "city": "Los Angeles", "region": "West"},
    {"name": "Bay Area Hub", "city": "San Francisco", "region": "West"},
    {"name": "Southern Charm", "city": "Atlanta", "region": "South"},
    {"name": "Lone Star", "city": "Houston", "region": "South"},
    {"name": "Cascade Corner", "city": "Seattle", "region": "West"},
    {"name": "Liberty Square", "city": "Philadelphia", "region": "East"},
    {"name": "Mile High Shop", "city": "Denver", "region": "Central"},
]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def build_products_df() -> pd.DataFrame:
    """Build products DataFrame with unique IDs."""
    rows = []
    for i, prod in enumerate(PRODUCTS, start=1):
        rows.append(
            {
                "product_id": i,
                "product_name": prod["name"],
                "category": prod["category"],
                "base_price": prod["base_price"],
            }
        )
    return pd.DataFrame(rows)


def build_stores_df() -> pd.DataFrame:
    """Build stores DataFrame with unique IDs."""
    rows = []
    for i, store in enumerate(STORES, start=1):
        rows.append(
            {
                "store_id": i,
                "store_name": store["name"],
                "city": store["city"],
                "region": store["region"],
            }
        )
    return pd.DataFrame(rows)


def build_sales_df(
    products_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    rng: random.Random,
) -> pd.DataFrame:
    """Generate sales transactions with realistic patterns.

    Adds seasonal variation (Q4 boost for electronics/clothing),
    weekend effects, and random quantity/discount distributions.
    """
    product_ids = products_df["product_id"].tolist()
    store_ids = stores_df["store_id"].tolist()
    price_lookup = dict(
        zip(products_df["product_id"], products_df["base_price"], strict=True)
    )
    category_lookup = dict(
        zip(products_df["product_id"], products_df["category"], strict=True)
    )

    date_range = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    dates = date_range.tolist()

    rows: list[dict] = []
    for sale_id in range(1, NUM_SALES + 1):
        product_id = rng.choice(product_ids)
        store_id = rng.choice(store_ids)
        sale_date = rng.choice(dates)
        category = category_lookup[product_id]

        # Seasonal multiplier: Q4 boost for Electronics and Clothing
        month = sale_date.month
        seasonal_mult = 1.0
        if month in (11, 12) and category in ("Electronics", "Clothing"):
            seasonal_mult = 1.5
        elif month in (6, 7) and category == "Food":
            seasonal_mult = 1.3

        # Quantity: 1-5 for most items, higher for food/cheap items
        base_price = price_lookup[product_id]
        max_qty = 3 if base_price > 100 else 8
        quantity = rng.randint(1, max_qty)

        # Discount: 0-30%, more likely to be 0
        discount_pct = rng.choice([0, 0, 0, 0, 5, 10, 10, 15, 20, 25, 30])

        unit_price = round(base_price * seasonal_mult, 2)
        total_amount = round(unit_price * quantity * (1 - discount_pct / 100), 2)

        rows.append(
            {
                "sale_id": sale_id,
                "sale_date": sale_date.strftime("%Y-%m-%d"),
                "product_id": product_id,
                "store_id": store_id,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount_pct": discount_pct,
                "total_amount": total_amount,
            }
        )

    df = pd.DataFrame(rows)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def write_csv_files(
    products_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    sales_df: pd.DataFrame,
) -> None:
    """Write all DataFrames as CSV files."""
    products_df.to_csv(DATA_DIR / "products.csv", index=False)
    stores_df.to_csv(DATA_DIR / "stores.csv", index=False)
    sales_df.to_csv(DATA_DIR / "sales.csv", index=False)
    print(f"  CSV: products.csv ({len(products_df)} rows)")
    print(f"  CSV: stores.csv ({len(stores_df)} rows)")
    print(f"  CSV: sales.csv ({len(sales_df)} rows)")


def write_parquet_files(
    products_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    sales_df: pd.DataFrame,
) -> None:
    """Write all DataFrames as Parquet files (single + partitioned)."""
    # Single Parquet files
    products_df.to_parquet(DATA_DIR / "products.parquet", index=False)
    stores_df.to_parquet(DATA_DIR / "stores.parquet", index=False)
    sales_df.to_parquet(DATA_DIR / "sales.parquet", index=False)
    print(f"  Parquet: products.parquet ({len(products_df)} rows)")
    print(f"  Parquet: stores.parquet ({len(stores_df)} rows)")
    print(f"  Parquet: sales.parquet ({len(sales_df)} rows)")

    # Year-partitioned Parquet files (for glob pattern exercises)
    partitioned_dir = DATA_DIR / "sales_by_year"
    partitioned_dir.mkdir(exist_ok=True)

    sales_df_copy = sales_df.copy()
    sales_df_copy["year"] = sales_df_copy["sale_date"].dt.year

    for year, group in sales_df_copy.groupby("year"):
        year_file = partitioned_dir / f"sales_{year}.parquet"
        group.drop(columns=["year"]).to_parquet(year_file, index=False)
        print(f"  Parquet: sales_by_year/sales_{year}.parquet ({len(group)} rows)")

    # Also write as Arrow table for exercise variety
    arrow_table = pa.Table.from_pandas(sales_df)
    pq.write_table(arrow_table, DATA_DIR / "sales_arrow.parquet")
    print(f"  Parquet (Arrow): sales_arrow.parquet ({len(sales_df)} rows)")


def generate_all() -> None:
    """Generate all sample data files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    print("Generating sample data...")

    products_df = build_products_df()
    stores_df = build_stores_df()
    sales_df = build_sales_df(products_df, stores_df, rng)

    print(f"\nDataset summary:")
    print(f"  Products: {len(products_df)} ({len(CATEGORIES)} categories)")
    print(f"  Stores: {len(stores_df)} ({len(set(s['region'] for s in STORES))} regions)")
    print(f"  Sales: {len(sales_df)} transactions (2022-2024)")
    print(f"  Date range: {sales_df['sale_date'].min()} to {sales_df['sale_date'].max()}")
    print()

    print("Writing CSV files...")
    write_csv_files(products_df, stores_df, sales_df)
    print()

    print("Writing Parquet files...")
    write_parquet_files(products_df, stores_df, sales_df)
    print()

    # Summary stats
    csv_size = sum(f.stat().st_size for f in DATA_DIR.glob("*.csv"))
    pq_size = sum(f.stat().st_size for f in DATA_DIR.glob("*.parquet"))
    pq_size += sum(
        f.stat().st_size for f in (DATA_DIR / "sales_by_year").glob("*.parquet")
    )

    print(f"Total CSV size: {csv_size / 1024 / 1024:.1f} MB")
    print(f"Total Parquet size: {pq_size / 1024 / 1024:.1f} MB")
    print(f"Compression ratio: {csv_size / pq_size:.1f}x")
    print()
    print("Data generation complete. Ready for exercises.")


if __name__ == "__main__":
    generate_all()
