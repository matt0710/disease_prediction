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

# def logistic_regression_model(stringIndexer, assembler, features_scaler, featureIndexer, trainingData, testData, evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")):
#
#      lr = LogisticRegression(featuresCol='indexedFeatures', labelCol='label')
#
#      paramGrid = ParamGridBuilder().addGrid(lr.elasticNetParam, [0.1, 0.3, 0.5, 0.8]) \
#          .addGrid(lr.maxIter, [100, 200, 500, 1000]) \
#          .addGrid(lr.regParam, [0.1, 0.3, 0.5, 0.8]).build()
#
#      pipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, lr])
#
#      tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)
#
#      model = tvs.fit(trainingData)
#
#      prediction = model.transform(testData)
#
#      accuracy = evaluator.evaluate(prediction)
#
#      return prediction, accuracy

