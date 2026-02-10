"""Spark session factory.

Creates a SparkSession configured for local development.
In production you'd connect to a cluster via spark-master URL instead.
"""

from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "RetailPipeline") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
