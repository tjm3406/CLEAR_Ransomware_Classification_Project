import os, sys
import argparse
import re
import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import matplotlib.pyplot as plt

FEATURE_COLS = [
    "is_write", "is_read",
    "Offset (Bytes)", "log_offset",
    "Size (Bytes)",   "log_size",
    "WaR", "RaR", "RaW", "WaW",
    "WaR Lapse", "RaR Lapse", "RaW Lapse", "WaW Lapse",
    "write_read_ratio", "total_overlap", "total_lapse",
]

def print_feature_map():
    print("\nFeature Index Mapping")
    for i, name in enumerate(FEATURE_COLS):
        print(f"\tfeature {i} = {name}")

def analyze_gbt(model, num_trees: int = 3):
    gbt_model = model.stages[-1]

    print(f"\tTotal trees:  {gbt_model.getNumTrees}")
    print(f"\tTotal nodes:  {gbt_model.totalNumNodes}")

    print_feature_map()

    # print(f"\nFirst {num_trees} Trees (of {gbt_model.getNumTrees})")
    # actual tree
    # for i, tree in enumerate(gbt_model.trees[:num_trees]):
    #     print(f"  Tree {i}  |  nodes: {tree.numNodes}  |  depth: {tree.depth}")
    #     print(tree.toDebugString)

    # feature importances 
    print(f"\nFeature importances:")
    importances = gbt_model.featureImportances.toArray()
    print(importances)
    plt.barh(FEATURE_COLS, importances)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("MLTraining") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "22g") \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.broadcast.blockSize", "8m")  \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    input_path = sys.argv[1]


    print(f"Loading model from: {input_path}")
    model = PipelineModel.load(input_path)
    analyze_gbt(model, num_trees=3)