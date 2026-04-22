import os, sys
os.environ.pop('SPARK_HOME', None)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

PYSPARK_HOME = r"C:\Users\sarah\miniforge3\envs\pyspark_env\Lib\site-packages\pyspark"

os.environ["SPARK_HOME"]  = PYSPARK_HOME
os.environ["HADOOP_HOME"] = PYSPARK_HOME  
os.environ["PATH"] = r"C:\Users\sarah\miniforge3\envs\pyspark_env\Lib\site-packages\pyspark\bin" + os.pathsep + os.environ["PATH"]
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


# Models to try

def build_logistic_regression():
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="Label",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0,   # L2 ridge; try 1.0 for Lasso
        family="binomial",
    )
    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.05, 0.1, 0.15]) # [0.001, 0.01, 0.1]
        .addGrid(lr.elasticNetParam, [0.25, 0.5, 0.75]) # [0, 0.5]
        .build()
    )
    return lr, param_grid


def build_linear_svm():
    svm = LinearSVC(
        featuresCol="features",
        labelCol="Label",
        maxIter=100,
        regParam=0.01,
    )
    param_grid = (
        ParamGridBuilder()
        .addGrid(svm.regParam, [0.05, 0.1, 0.15]) # [0.001, 0.01, 0.1]
        .addGrid(svm.maxIter,  [100, 150]) # [50, 100]
        .build()
    )
    return svm, param_grid

 
def build_decision_tree():
    dt = DecisionTreeClassifier(
        featuresCol="features",
        labelCol="Label",
        maxDepth=8,
        seed=42,
    )
    param_grid = (
        ParamGridBuilder()
        .addGrid(dt.maxDepth, [9, 10, 11, 12]) # [5, 8, 10]
        .build()
    )
    return dt, param_grid


def build_random_forest():
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="Label",
        numTrees=100,
        maxDepth=10,
        seed=42,
        # Class imbalance: weight minority class higher
        # (set after checking balance — uncomment if ransomware is minority)
        # weightCol="class_weight",
    )
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.numTrees,  [50, 100])
        .addGrid(rf.maxDepth,  [8, 12])
        .build()
    )
    return rf, param_grid

 
def build_gbt():
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="Label",
        maxIter=50,
        maxDepth=8,
        stepSize=0.1,
        seed=42,
    )
    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxIter,  [30, 50])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .build()
    )
    return gbt, param_grid


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

# training with cross val, default 3 fold
def train_with_cv(
    name: str,
    classifier,
    param_grid,
    train_df,
    cv_folds: int = 3,
):
    # pipeline for cv
    pipeline = Pipeline(stages=[classifier])
 
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="Label",
        metricName="areaUnderROC",
    )
 
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=auc_evaluator,
        numFolds=cv_folds,
        seed=42,
        parallelism=2,
    )
 
    print(f"\nTraining {name} (CV={cv_folds})")
    cv_model = cv.fit(train_df)
    best_model = cv_model.bestModel
    check_convergence(name, best_model)
    best_classifier = best_model.stages[-1]
    print(f"\tBest CV AUC: {max(cv_model.avgMetrics):.4f}")
    print(f"\tBest params:")
    for param, value in best_classifier.extractParamMap().items():
        print(f"\t{param.name:<30} = {value}")
    return best_model

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
    f1        = mc_eval.setMetricName("f1").evaluate(predictions)
    precision = mc_eval.setMetricName("weightedPrecision").evaluate(predictions)
    recall    = mc_eval.setMetricName("weightedRecall").evaluate(predictions)

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

# check class balance as percentage
def check_class_balance(df):
    total = df.count()
    df.groupBy("Label").count() \
      .withColumn("pct", F.round(F.col("count") / total * 100, 2)) \
      .show()

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
    # scaler = StandardScaler(
    #             inputCol="raw_features",
    #             outputCol="features",
    #             withMean=True,
    #             withStd=True,
    #         )
    # scaler_model = scaler.fit(raw_training_data)
    # scaled_train_data = scaler_model.transform(raw_training_data)
    # scaled_test_data = scaler_model.transform(raw_test_data)
    # print("Data scaled")
    train_data = raw_training_data
    test_data = raw_test_data
    # print("Training data preview...")
    # train_data.show()
    # print("Caching...")
    # scaled_train_data.cache()
    # scaled_test_data.cache()
    # print(f"\tTrain: {scaled_train_data.count():,}  |  Test: {scaled_test_data.count():,}")


    # 3. setup models
    print("Model setup")
    models_cfg = [
        ("Logistic Regression", *build_logistic_regression()),
        ("Linear SVM",          *build_linear_svm()),
        ("Decision Tree",       *build_decision_tree()),
        ("Random Forest",       *build_random_forest()),
        ("GBT",                 *build_gbt()),
    ]

    # train + eval + save
    results = []
    for name, classifier, param_grid in models_cfg:
        best_model = train_with_cv(
            name, classifier, param_grid, train_data, cv_folds=3
        )
        metrics = evaluate_model(name, best_model, test_data)
        results.append(metrics)
        path = os.path.join(output_path, name.replace(" ", "_"))
        best_model.write().overwrite().save(path)
        print(f"\tSaved {name} → {path}")

    print_summary(results) 