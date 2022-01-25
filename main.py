from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, NaiveBayes, MultilayerPerceptronClassifier, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from utils import *

spark = SparkSession.builder.appName("simpleapp").getOrCreate()

trainingData, names = preprocess_data("Training.csv", spark)
testData, _ = preprocess_data("Testing.csv", spark)

stringIndexer = StringIndexer(inputCol='prognosis', outputCol='label')
assembler = VectorAssembler(inputCols=[x for x in names[:-1]], outputCol='features')
features_scaler = MinMaxScaler(inputCol="features", outputCol="sfeatures")
featureIndexer = VectorIndexer(inputCol="sfeatures", outputCol="indexedFeatures", maxCategories=41)

#_______________________________________________________________________________________________________________________

lr = LogisticRegression(featuresCol='indexedFeatures', labelCol='label')


paramGrid = ParamGridBuilder().addGrid(lr.elasticNetParam, [0.1, 0.3, 0.5, 0.8])\
    .addGrid(lr.maxIter, [100, 200, 500, 1000])\
    .addGrid(lr.regParam, [0.1, 0.3, 0.5, 0.8]).build()

pipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, lr])

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)

model = tvs.fit(trainingData)

lrPrediction = model.transform(testData)

lrPrediction.show()
lrAccuracy = evaluator.evaluate(lrPrediction)
print(lrAccuracy)

#_______________________________________________________________________________________________________________________

nb = NaiveBayes(modelType="multinomial")

paramGrid = ParamGridBuilder().addGrid(nb.modelType, ["multinomial", "gaussian", "complement"])\
    .addGrid(nb.smoothing, [0.8, 0.9, 1]).build()

nbPipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, nb])
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
nbTvs = TrainValidationSplit(estimator=nbPipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)
nbModel = nbTvs.fit(trainingData)
nbPrediction = nbModel.transform(testData)
nbPrediction.show()
nbAccuracy = evaluator.evaluate(nbPrediction)
print("Naive Bayes accuracy: ", nbAccuracy)

#_______________________________________________________________________________________________________________________


layers = [130,50,25,41]

mlp = MultilayerPerceptronClassifier(layers=layers) # maxIter=100, layers=layers, blockSize=128, seed=1234)
paramGrid = ParamGridBuilder().addGrid(mlp.maxIter, [100, 500, 1000])\
    .addGrid(mlp.blockSize, [64, 128, 192])\
    .addGrid(mlp.seed, [1234, 5678, 2468]).build()

mlpPipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, mlp])
mlpTvs = TrainValidationSplit(estimator=mlpPipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)
mlpModel = mlpTvs.fit(trainingData)
mlpPrediction = mlpModel.transform(testData)
mlpPrediction.show()
mlpAccuracy = evaluator.evaluate(mlpPrediction)
print("MLP accuracy: ", mlpAccuracy)

#_______________________________________________________________________________________________________________________

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dtPipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, dt])
dt_model = dtPipeline.fit(trainingData)
dt_predictions = dt_model.transform(testData)
dt_predictions.show()
dtAccuracy = evaluator.evaluate(dt_predictions)

print("Decision Tree accuracy: ", dtAccuracy)

rt = RandomForestClassifier(labelCol="label", featuresCol="features")
paramGrid = ParamGridBuilder().addGrid(rt.numTrees, [100, 500, 1000])\
    .addGrid(rt.maxDepth, [5, 10, 15])\
    .addGrid(rt.seed, [1234, 5678, 2468]).build()

rtPipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, rt])
rtTvs = TrainValidationSplit(estimator=rtPipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)
rtModel = rtTvs.fit(trainingData)
rtPrediction = rtModel.transform(testData)
rtPrediction.show()
rtAccuracy = evaluator.evaluate(rtPrediction)
print("RT accuracy: ", rtAccuracy)