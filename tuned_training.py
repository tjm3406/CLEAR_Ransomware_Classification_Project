import os, sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import (
    LogisticRegression,
    LinearSVC,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics

# models for cluster (hardcoded best params from local CV tuning)

def build_lr_cluster(
    reg_param: float      = 0.1,
    elastic_net: float    = 0.5,
    max_iter: int         = 100,
):
    return LogisticRegression(
        featuresCol="features",
        labelCol="Label",
        regParam=reg_param,
        elasticNetParam=elastic_net,
        maxIter=max_iter,
        family="binomial",  
    )

def build_svm_cluster(
    reg_param: float = 0.1,
    max_iter: int    = 100,
):
    return LinearSVC(
        featuresCol="features",
        labelCol="Label",
        regParam=reg_param,
        maxIter=max_iter,
    )

def build_dt_cluster(
    max_depth: int = 10,
):
    return DecisionTreeClassifier(
        featuresCol="features",
        labelCol="Label",
        maxDepth=max_depth,
        seed=42,
    )

def build_rf_cluster(
    num_trees: int = 75,
    max_depth: int = 12,
):
    return RandomForestClassifier(
        featuresCol="features",
        labelCol="Label",
        numTrees=num_trees,
        maxDepth=max_depth,
        seed=42,
    )

def build_gbt_cluster(
    max_iter: int      = 50,
    max_depth: int     = 8,
    step_size: float   = 0.5,
):
    return GBTClassifier(
        featuresCol="features",
        labelCol="Label",
        maxIter=max_iter,
        maxDepth=max_depth,
        stepSize=step_size,
        seed=42,
    )


# check if model actually converged
def check_convergence(name: str, model):
    classifier = model.stages[-1]

    if name == "Linear SVM":    # can't show convergence builtin
        print(f"\tN/A")
    
    elif name == "Logistic Regression":
        summary = classifier.summary
        max_iter = classifier.getMaxIter()
        total_iter = summary.totalIterations
        converged = total_iter < max_iter

        print(f"\tConverged: {converged}")
        print(f"\tIterations: {total_iter} / {max_iter}")

        if hasattr(summary, "objectiveHistory"):
            history = summary.objectiveHistory
            print(f"\tFinal loss: {history[-1]:.6f}")
            print(f"\tLoss improvement: {history[-2] - history[-1]:.8f}" if len(history) > 1 else "")

        if not converged:
            print(f"\tHit maxIter - did not converge")
    elif name == "GBT":
        print(f"\tTrees built: {classifier.getNumTrees}")
        print(f"\tTotal nodes: {classifier.totalNumNodes}")
    elif name == "Random Forest":
        print(f"\tTrees built: {classifier.getNumTrees}")
        print(f"\tTotal nodes: {classifier.totalNumNodes}")
    elif name == "Decision Tree":
        print(f"\tTotal nodes: {classifier.numNodes}")
        print(f"\tDepth: {classifier.depth}")
    else:
        print(f"\tN/A")

def train_single(
    name: str,
    classifier,
    train_df,
):
    # pipeline for cv
    pipeline = Pipeline(stages=[classifier])
    print(f"\nTraining {name}")
    model = pipeline.fit(train_df)
    check_convergence(name, model)
    print(f"Done.")
    return model


# evaluate, prints AUC-ROC, AUC-PR, accuracy, F1, precision, recall, and confusion matrix
def evaluate_model(name: str, model, test_df):
    predictions = model.transform(test_df)

    auc_roc = BinaryClassificationEvaluator(
        labelCol="Label", metricName="areaUnderROC"
    ).evaluate(predictions)

    auc_pr = BinaryClassificationEvaluator(
        labelCol="Label", metricName="areaUnderPR"
    ).evaluate(predictions)

    mc_eval = MulticlassClassificationEvaluator(
        labelCol="Label", predictionCol="prediction"
    )
    accuracy  = mc_eval.setMetricName("accuracy").evaluate(predictions)
    f1        = mc_eval.setMetricName("fMeasureByLabel").evaluate(predictions)
    precision = mc_eval.setMetricName("precisionByLabel").evaluate(predictions)
    recall    = mc_eval.setMetricName("recallByLabel").evaluate(predictions)

    print(f"\tResults — {name}")
    print(f"\tAUC-ROC   : {auc_roc:.4f}")
    print(f"\tAUC-PR    : {auc_pr:.4f}")
    print(f"\tAccuracy  : {accuracy:.4f}")
    print(f"\tF1 (wtd)  : {f1:.4f}")
    print(f"\tPrecision : {precision:.4f}")
    print(f"\tRecall    : {recall:.4f}")

    # Confusion matrix via RDD API
    pred_and_labels = (
        predictions.select("prediction", "Label")
        .rdd
        .map(lambda r: (float(r["prediction"]), float(r["Label"])))
    )
    cm = MulticlassMetrics(pred_and_labels)
    print(f"\n\tConfusion Matrix (rows=actual, cols=predicted):")
    print(f"\tTN={cm.confusionMatrix().toArray()[0][0]:.0f}  "
          f"FP={cm.confusionMatrix().toArray()[0][1]:.0f}")
    print(f"\tFN={cm.confusionMatrix().toArray()[1][0]:.0f}  "
          f"TP={cm.confusionMatrix().toArray()[1][1]:.0f}")

    return {
        "name": name,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def print_summary(results: list):
    print("\n\nSummary Table")
    print(f"{'Model':<25} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Accuracy':>10}")
    print("-" * 63)
    for r in sorted(results, key=lambda x: x["auc_roc"], reverse=True):
        print(
            f"{r['name']:<25} {r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f}"
            f" {r['f1']:>8.4f} {r['accuracy']:>10.4f}"
        )
    print()


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
    output_path = sys.argv[2]

    # 1. inital stuff, load data, see data
    print("Reading data...")
    df = spark.read.parquet(input_path)
    print("Data read")
    # print("Class balance:")
    # check_class_balance(df)

    # df.show()

    features = [
        "Timestamp",
        "is_write", "is_read",
        "Offset (Bytes)", "log_offset",
        "Size (Bytes)", "log_size",
        "WaR", "RaR", "RaW", "WaW",
        "WaR Lapse", "RaR Lapse", "RaW Lapse", "WaW Lapse",
        "write_read_ratio", "total_overlap", "total_lapse",
        "Process ID",
    ]
    
    print("Transforming data...")
    assembler = VectorAssembler(
                    inputCols=features, 
                    outputCol='features' # raw if gonna go through scaler
                )
    transformed_data = assembler.transform(df)
    print("Transformed")
    print("Split and scale...")
    # 2. data split and scale (needed for logistic regression)
    (raw_training_data, raw_test_data) = transformed_data.randomSplit([0.8, 0.2], 42) # seeded for consistency while testing 
    print("Data split")

    train_data = raw_training_data
    test_data = raw_test_data
    print("Training data preview...")
    # train_data.show()

    print("Model setup")
    models_cfg = [
        # ("Logistic Regression", build_lr_cluster()),
        # ("Linear SVM",          build_svm_cluster()),
        # ("Decision Tree",       build_dt_cluster()),
        # ("Random Forest",       build_rf_cluster()),
        ("GBT",                 build_gbt_cluster()),
    ]

    results = []
    for name, classifier in models_cfg:
        best_model = train_single(
            name, classifier, train_data
        )
        metrics = evaluate_model(name, best_model, test_data)
        results.append(metrics)
        
        path = os.path.join(output_path, name.replace(" ", "_"))
        best_model.write().overwrite().save(path)
        print(f"\tSaved {name} to {path}")

    print_summary(results)