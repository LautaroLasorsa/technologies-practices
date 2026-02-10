"""Phase 1 -- Verify that the local driver can reach the Docker Spark master."""

from spark_helpers import create_spark_session


def main() -> None:
    spark = create_spark_session("connectivity-check")
    print(f"Spark version : {spark.version}")
    print(f"Master        : {spark.sparkContext.master}")
    print(f"App name      : {spark.sparkContext.appName}")
    print(f"Default parallelism: {spark.sparkContext.defaultParallelism}")
    print("Connectivity check passed.")
    spark.stop()


if __name__ == "__main__":
    main()
