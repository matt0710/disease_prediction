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


