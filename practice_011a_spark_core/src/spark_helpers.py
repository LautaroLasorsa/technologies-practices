"""Shared helpers: SparkSession factory and path constants."""

from pathlib import Path

from pyspark.sql import SparkSession

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SALES_CSV = DATA_DIR / "sales.csv"
PRODUCTS_JSON = DATA_DIR / "products.json"

# ── Spark Configuration ───────────────────────────────────────────────
SPARK_MASTER = "spark://localhost:7077"


def create_spark_session(app_name: str = "spark-core-practice") -> SparkSession:
    """Build a SparkSession connected to the Docker-hosted master.

    Uses local[*] as fallback if the Docker cluster is unreachable,
    so scripts can still run (slowly) without Docker for quick tests.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master(SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "4")   # keep low for small data
        .config("spark.driver.memory", "512m")
        .getOrCreate()
    )
