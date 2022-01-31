import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, NaiveBayes, MultilayerPerceptronClassifier, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def process_data(file, spark):
    df_1 = spark.read.csv(file, header=True, inferSchema=True)

    df_2 = df_1.drop("fluid_overload117")
    df_3 = df_2.drop("fluid_overload45")
    df_4 = df_3.drop("_c133")

    df_4.printSchema()

    names = df_4.columns

    indexer = StringIndexer(inputCol='prognosis', outputCol='label').fit(df_4)

    indexer.transform(df_4).show(5, True)

    assembler = VectorAssembler(inputCols=[x for x in names[:-1]], outputCol='features').transform(
        indexer.transform(df_4))
    assembler.show()

    features_scaler = MinMaxScaler(inputCol="features", outputCol="sfeatures")
    smodel = features_scaler.fit(assembler).transform(assembler)

    featureIndexer = VectorIndexer(inputCol="sfeatures", \
                                   outputCol="indexedFeatures", \
                                   maxCategories=41).fit(smodel)

    trainingData = featureIndexer.transform(smodel)

    return trainingData


def preprocess_data(file, spark):

    df_1 = spark.read.csv(file, header=True, inferSchema=True)

    df_2 = df_1.drop("fluid_overload117")
    df_3 = df_2.drop("fluid_overload45")
    df_4 = df_3.drop("_c133")

    names = df_4.columns

    return df_4, names


def evaluate_model(evaluator, model):

    evaluator.setMetricName("accuracy")
    accuracy = evaluator.evaluate(model)

    evaluator.setMetricName("hammingLoss")
    hammingLoss = evaluator.evaluate(model)

    evaluator.setMetricName("precisionByLabel")
    precisionByLabel = evaluator.evaluate(model)

    evaluator.setMetricName("recallByLabel")
    recallByLabel = evaluator.evaluate(model)

    evaluator.setMetricName("logLoss")
    logLoss = evaluator.evaluate(model)

    return accuracy, hammingLoss, precisionByLabel, recallByLabel, logLoss


def three_dim_plot(p1, p2, p3, acc, name1, name2, name3, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z, c = np.array(p1), np.array(p2), np.array(p3), np.array(acc)
    img = ax.scatter(x, y, z, c=c)
    fig.colorbar(img)
    plt.legend((x, y, z), (str(name1), str(name2), str(name3)))
    ax.set_xlabel(str(name1)), ax.set_ylabel(str(name2)), ax.set_zlabel(str(name3))
    ax.set_title(str(title))
    plt.show()


def print_heatmap(matrix, x, y, title, size):
    fig, ax = plt.subplots(figsize=size)
    ax = sns.heatmap(matrix, cmap="YlGnBu")
    ax.xaxis.set_ticklabels(x, rotation=270)
    ax.yaxis.set_ticklabels(y, rotation=0)
    ax.set_title(str(title))
    plt.tight_layout()
    plt.show()


def trees_feature_importance(feature, importance, title):
    fig, ax = plt.subplots(figsize=(13, 13))

    plt.bar(feature, importance, orientation='vertical', width=0.4)
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.xticks(range(importance.shape[0]), feature, rotation=90, fontsize=8)
    ax.set_title(str(title))

    plt.show()
