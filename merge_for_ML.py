import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("PrepareML") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.memoryOverhead", "512m") \
        .config("spark.driver.maxResultSize", "512m") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.files.maxPartitionBytes", "64mb") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .getOrCreate()

    input_path = sys.argv[1]  # gs://.../data_to_use/
    output_path = sys.argv[2]  # gs://.../ml_ready

    df = spark.read.parquet(f"{input_path}/batch_*")

    # Drop columns that could cause data leakage or noise
    df = df.drop("Process Name", "Path Name")

    # Drop nulls
    df = df.dropna()

    # Feature engineering
    # includes making new features or making features more usable such as scaling the size and offset in bytes
    df = df \
        .withColumn("is_write", (F.col("OpCode") == 1).cast(LongType())) \
        .withColumn("is_read",  (F.col("OpCode") == 2).cast(LongType())) \
        .withColumn("write_read_ratio",
            F.when(F.col("Size (Bytes)") > 0,
                F.col("WaR") / F.col("Size (Bytes)"))
            .otherwise(0.0)) \
        .withColumn("total_overlap",
            F.col("WaR") + F.col("RaR") + F.col("RaW") + F.col("WaW")) \
        .withColumn("total_lapse",
            F.col("WaR Lapse") + F.col("RaR Lapse") +
            F.col("RaW Lapse") + F.col("WaW Lapse")) \
        .withColumn("log_size",   F.log1p(F.col("Size (Bytes)"))) \
        .withColumn("log_offset", F.log1p(F.col("Offset (Bytes)")))

    # Assemble final feature columns
    feature_cols = [
        "Timestamp",
        "is_write", "is_read",
        "Offset (Bytes)", "log_offset",
        "Size (Bytes)",   "log_size",
        "WaR", "RaR", "RaW", "WaW",
        "WaR Lapse", "RaR Lapse", "RaW Lapse", "WaW Lapse",
        "write_read_ratio", "total_overlap", "total_lapse",
        "Process ID",
        "Label",
    ]

    df = df.select(feature_cols)

    # Write output partitioned by Label
    df.repartition(200) \
      .write \
      .mode("overwrite") \
      .option("maxRecordsPerFile", 500000) \
      .partitionBy("Label") \
      .parquet(output_path)

    print(f"ML-ready data written to {output_path}")