import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

LABEL_COL = "Label"


def show_counts(label: str, df):
    total = df.count()
    print(f"\n{label}")
    df.groupBy(LABEL_COL).count() \
      .withColumn("pct", F.round(F.col("count") / total * 100, 2)) \
      .orderBy(LABEL_COL) \
      .show()
    print(f"   Total rows: {total:,}\n")


def stratified_sample(df, fraction: float, seed: int):
    """
    Sample `fraction` of rows from each class independently,
    preserving the original class ratio.
    """
    fractions = {
        row[LABEL_COL]: fraction
        for row in df.select(LABEL_COL).distinct().collect()
    }
    return df.sampleBy(LABEL_COL, fractions=fractions, seed=seed)


def downsample(input_path: str, output_path: str, fraction: float, seed: int):
    spark = SparkSession.builder \
            .appName("dwnsample") \
            .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    print(f"\nLoading: {input_path}")
    df = spark.read.parquet(input_path)

    # show_counts("Original", df)
    fraction = float(fraction) # typcasting issue
    sampled = stratified_sample(df, fraction, seed)
    sampled.persist()

    show_counts(f"Sampled ({fraction*100:.1f}% per class)", sampled)

    print(f"Saving to: {output_path}")
    sampled.repartition(4).write.mode("overwrite").parquet(output_path)
    print("Done.\n")

    spark.stop()


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    percent_to_save = sys.argv[3] # as decimal, maybe 0.05 0.00001% = 0.0000001


    downsample(input_path, output_path, percent_to_save, 42)