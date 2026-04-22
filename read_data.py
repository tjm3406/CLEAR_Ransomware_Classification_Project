import sys
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import LongType, StringType
from google.cloud import storage
from functools import reduce

SETS = {'Set 1', 'Set 2'}
BATCH_SIZE = 50

CHECKPOINT_FILE = "gs://metcs777-term-project-clear/data_to_use/checkpoint.txt"
VALID_PATHS_FILE = "gs://metcs777-term-project-clear/data_to_use/valid_paths.txt"

# All column name variants
COLUMN_ALIASES = {
    "Offset":    "Offset (Bytes)",
    "Size":      "Size (Bytes)",
    "WAR":       "WaR",
    "RAR":       "RaR",
    "RAW":       "RaW",
    "WAW":       "WaW",
    "WAR Lapse": "WaR Lapse",
    "RAR Lapse": "RaR Lapse",
    "RAW Lapse": "RaW Lapse",
    "WAW Lapse": "WaW Lapse",
    "Path":      "Path Name",
}

STRING_COLUMNS = {"Process Name", "Path Name", "Path"}


def get_gcs_client():
    return storage.Client()


def parse_gcs_path(gcs_path: str):
    assert gcs_path.startswith("gs://"), f"Expected gs:// path, got: {gcs_path}"
    without_scheme = gcs_path[5:]
    bucket, _, prefix = without_scheme.partition("/")
    return bucket, prefix


def load_valid_paths() -> list:
    try:
        client = get_gcs_client()
        bucket_name, blob_path = parse_gcs_path(VALID_PATHS_FILE)
        blob = client.bucket(bucket_name).blob(blob_path)
        paths = blob.download_as_text().strip().splitlines()
        print(f"Loaded {len(paths)} valid paths from cache")
        return paths
    except:
        return []


def save_valid_paths(paths: list):
    try:
        client = get_gcs_client()
        bucket_name, blob_path = parse_gcs_path(VALID_PATHS_FILE)
        blob = client.bucket(bucket_name).blob(blob_path)
        blob.upload_from_string("\n".join(paths))
        print(f"Saved {len(paths)} valid paths to cache")
    except Exception as e:
        print(f"Warning: could not save valid paths: {e}")


def load_checkpoint() -> int:
    try:
        client = get_gcs_client()
        bucket_name, blob_path = parse_gcs_path(CHECKPOINT_FILE)
        blob = client.bucket(bucket_name).blob(blob_path)
        return int(blob.download_as_text().strip())
    except:
        return -1


def save_checkpoint(batch_index: int):
    try:
        client = get_gcs_client()
        bucket_name, blob_path = parse_gcs_path(CHECKPOINT_FILE)
        blob = client.bucket(bucket_name).blob(blob_path)
        blob.upload_from_string(str(batch_index))
    except Exception as e:
        print(f"Warning: could not save checkpoint: {e}")


def list_subfolders(root_path: str) -> list:
    client = get_gcs_client()
    bucket_name, root_prefix = parse_gcs_path(root_path)
    folders = []

    for label in ["Benign", "Ransomware"]:
        prefix = f"{root_prefix}{label}/"
        blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
        list(blobs)
        for subfolder_prefix in blobs.prefixes:
            folders.append(f"gs://{bucket_name}/{subfolder_prefix}")

    return folders


def read_metadata_gcs(gcs_path: str) -> bool:
    try:
        client = get_gcs_client()
        bucket_name, blob_path = parse_gcs_path(gcs_path)
        blob = client.bucket(bucket_name).blob(blob_path)
        content = blob.download_as_text()
        data = json.loads(content)
        return data.get("Victim Data") in SETS
    except Exception as e:
        print(f"Error reading metadata {gcs_path}: {e}")
        return False


#Read a single parquet file, cast all numerics to LongType, normalize column names
def read_and_normalize(path: str) -> DataFrame:
    df = spark.read \
        .option("spark.sql.parquet.enableVectorizedReader", "false") \
        .parquet(path)

    # Cast all columns to safe types
    for col_name in df.columns:
        if col_name in STRING_COLUMNS:
            df = df.withColumn(col_name, F.col(f"`{col_name}`").cast(StringType()))
        else:
            df = df.withColumn(col_name, F.col(f"`{col_name}`").cast(LongType()))

    # Normalize V2 column names to canonical V1 names
    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)

    return df


def process_batch(batch_paths: list, batch_index: int, output_path: str):
    print(f"Batch {batch_index}: reading {len(batch_paths)} files")

    dfs = []
    for path in batch_paths:
        try:
            dfs.append(read_and_normalize(path))
        except Exception as e:
            print(f"  Skipping {path}: {e}")

    if not dfs:
        save_checkpoint(batch_index)  # skip it and move on
        return

    batch_df = reduce(DataFrame.unionByName, dfs)
    batch_df.coalesce(50) \
        .write.mode("overwrite") \
        .option("maxRecordsPerFile", 500000) \
        .parquet(f"{output_path}/batch_{batch_index}")

    print(f"Batch {batch_index}")
    batch_df.unpersist()
    spark.catalog.clearCache()


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("MySparkApp") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.memoryOverhead", "512m") \
        .config("spark.driver.maxResultSize", "512m") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.files.maxPartitionBytes", "64mb") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.task.maxFailures", "10") \
        .getOrCreate()

    # Load or build valid_paths
    valid_paths = load_valid_paths()

    if not valid_paths:
        print("No cache found.")
        folders = list_subfolders(sys.argv[1])
        for folder in folders:
            try:
                folder_id = folder.rstrip("/").split("/")[-1]
                metadata_path = f"{folder}recording_{folder_id}_metadata.json"
                parquet_path = f"{folder}recording_{folder_id}.parquet"
                if read_metadata_gcs(metadata_path):
                    valid_paths.append(parquet_path)
                    print(f"APPROVE: {folder}")
                else:
                    print(f"SKIPPED: {folder}")
            except Exception as e:
                print(f"Error processing {folder}: {e}")
        save_valid_paths(valid_paths)

    if not valid_paths:
        print("No valid paths checkpoint found")
        sys.exit(0)

    batches = [valid_paths[i:i + BATCH_SIZE] for i in range(0, len(valid_paths), BATCH_SIZE)]
    print(f"Total files: {len(valid_paths)} across {len(batches)} batches")

    last_completed = load_checkpoint()
    start_batch = last_completed + 1
    print(f"Resuming from batch {start_batch}")

    for batch_index, batch_paths in enumerate(batches):
        if batch_index < start_batch:
            continue
        process_batch(batch_paths, batch_index, sys.argv[2])
        save_checkpoint(batch_index)

    print(f"All batches complete. Output at: {sys.argv[2]}")