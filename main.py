from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer, OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, NaiveBayes, MultilayerPerceptronClassifier, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.stat import Correlation
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import col

from utils import *

spark = SparkSession.builder.appName("disease_detection").getOrCreate()

trainingData, names = preprocess_data("Training.csv", spark)
testData, _ = preprocess_data("Testing.csv", spark)

stringIndexer = StringIndexer(inputCol='prognosis', outputCol='label')
assembler = VectorAssembler(inputCols=[x for x in names[:-1]], outputCol='features')
features_scaler = MinMaxScaler(inputCol="features", outputCol="sfeatures")
featureIndexer = VectorIndexer(inputCol="sfeatures", outputCol="indexedFeatures", maxCategories=41)

#_____________________________________________________________________

feature = names
feature.remove('prognosis')

assembler2 = VectorAssembler(inputCols=feature, outputCol='features')
train = assembler2.transform(trainingData)

matrix = Correlation.corr(train.select('features'), 'features')
matrix_np = matrix.collect()[0]["pearson({})".format('features')].values

matrix_np = matrix_np.reshape(len(feature), len(feature))

fig, ax = plt.subplots(figsize=(12, 8))
ax = sns.heatmap(matrix_np, cmap="YlGnBu")
ax.xaxis.set_ticklabels(feature, rotation=270)
ax.yaxis.set_ticklabels(feature, rotation=0)
ax.set_title("Pearson Correlation Matrix")
plt.tight_layout()
plt.show()

#________________________________________________________

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

lrAccuracy, lrHammingLoss, lrPrecision, lrRecall, lrLogLoss = evaluate_model(evaluator, lrPrediction)

print("Logistic Regression's measures: \n" + "\taccuracy: " + str(lrAccuracy) + \
      "\n\tHamming Loss: " + str(lrHammingLoss) + "\n\tPrecision By Label: " + str(lrPrecision) + \
      "\n\tRecall By Label: " + str(lrRecall) + "\n\tLog Loss: " + str(lrLogLoss))

lrBestPipeline = model.bestModel
lrBestModel = lrBestPipeline.stages[-1]

print("maxIter: ", lrBestModel.getOrDefault('maxIter'))
print("regParam: ", lrBestModel.getOrDefault('regParam'))
print("elasticNetParam: ", lrBestModel.getOrDefault('elasticNetParam'))

print(lrBestModel.coefficientMatrix)

lrMatrix = lrBestModel.coefficientMatrix

coefficientM = []

for i in range(41):
    coeffientList = []
    for j in range(130):
        coeffientList.append(lrMatrix[i, j])
    coefficientM.append(coeffientList)

print(coefficientM)
#___________________________________________________________________________

# nb = NaiveBayes(modelType="multinomial")
#
# paramGrid = ParamGridBuilder().addGrid(nb.modelType, ["multinomial", "gaussian", "complement"])\
#     .addGrid(nb.smoothing, [0.8, 0.9, 1]).build()
#
# nbPipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, nb])
# evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
# nbTvs = TrainValidationSplit(estimator=nbPipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)
# nbModel = nbTvs.fit(trainingData)
# nbPrediction = nbModel.transform(testData)
# nbPrediction.show()
#
# nbAccuracy, nbHammingLoss, nbPrecision, nbRecall, nbLogLoss = evaluate_model(evaluator, nbPrediction)
#
# print("Naive Bayes measures: \n" + "\taccuracy: " + str(nbAccuracy) + \
#       "\n\tHamming Loss: " + str(nbHammingLoss) + "\n\tPrecision By Label: " + str(nbPrecision) + \
#       "\n\tRecall By Label: " + str(nbRecall) + "\n\tLog Loss: " + str(nbLogLoss))
#
# nbBestPipeline = nbModel.bestModel
# nbBestModel = nbBestPipeline.stages[-1]
#
# print("model_type: ", nbBestModel.getModelType())
# print("smoothing: ", nbBestModel.getOrDefault('smoothing'))
#
# for item, acc in zip(nbModel.getEstimatorParamMaps(), nbModel.validationMetrics):
#     print("the smoothing is: " + str(item.values()[1]) + " while the model_type is: " + str(item.values()[0]) + " and the accuracy is: " + str(acc))

#___________________________________________________________________


layers = [130, 50, 25, 41]

mlp = MultilayerPerceptronClassifier(layers=layers)
paramGrid = ParamGridBuilder().addGrid(mlp.maxIter, [100, 500, 1000])\
    .addGrid(mlp.blockSize, [64, 128, 192])\
    .addGrid(mlp.seed, [1234, 5678, 2468]).build()

mlpPipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, mlp])
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
mlpTvs = TrainValidationSplit(estimator=mlpPipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)
mlpModel = mlpTvs.fit(trainingData)
mlpPrediction = mlpModel.transform(testData)
mlpPrediction.show()

mlpAccuracy, mlpHammingLoss, mlpPrecision, mlpRecall, mlpLogLoss = evaluate_model(evaluator, mlpPrediction)

print("MLP's measures: \n" + "\taccuracy: " + str(mlpAccuracy) + \
      "\n\tHamming Loss: " + str(mlpHammingLoss) + "\n\tPrecision By Label: " + str(mlpPrecision) + \
      "\n\tRecall By Label: " + str(mlpRecall) + "\n\tLog Loss: " + str(mlpLogLoss))


mlpBestPipeline = mlpModel.bestModel
mlpBestModel = mlpBestPipeline.stages[-1]

print("maxIter: ", mlpBestModel.getOrDefault('maxIter'))
print("blockSize: ", mlpBestModel.getOrDefault('blockSize'))
print("seed: ", mlpBestModel.getOrDefault('seed'))

for item, acc in zip(mlpModel.getEstimatorParamMaps(), mlpModel.validationMetrics):
    print("the max_iter is: " + str(item.values()[0]) + " while the block_size is: " + str(item.values()[1]) + \
          " while the seed is: " + str(item.values()[2])  +  " and the accuracy is: " + str(acc))

#____________________________________________________________

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
dtPipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, dt])
dt_model = dtPipeline.fit(trainingData)
dt_predictions = dt_model.transform(testData)
dt_predictions.show()
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

dtAccuracy, dtHammingLoss, dtPrecision, dtRecall, dtLogLoss = evaluate_model(evaluator, dt_predictions)

print("Decision Tree's measures: \n" + "\taccuracy: " + str(dtAccuracy) + \
      "\n\tHamming Loss: " + str(dtHammingLoss) + "\n\tPrecision By Label: " + str(dtPrecision) + \
      "\n\tRecall By Label: " + str(dtRecall) + "\n\tLog Loss: " + str(dtLogLoss))

#__________________________________________________________________________________________________________________

rt = RandomForestClassifier(labelCol="label", featuresCol="features")
paramGrid = ParamGridBuilder().addGrid(rt.numTrees, [100, 500, 1000])\
    .addGrid(rt.maxDepth, [5, 10, 15])\
    .addGrid(rt.seed, [1234, 5678, 2468]).build()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
rtPipeline = Pipeline(stages=[stringIndexer, assembler, features_scaler, featureIndexer, rt])
rtTvs = TrainValidationSplit(estimator=rtPipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, trainRatio=0.8)
rtModel = rtTvs.fit(trainingData)
rtPrediction = rtModel.transform(testData)
rtPrediction.show()

rtAccuracy, rtHammingLoss, rtPrecision, rtRecall, rtLogLoss = evaluate_model(evaluator, rtPrediction)

print("Random Forest's measures: \n" + "\taccuracy: " + str(rtAccuracy) + \
      "\n\tHamming Loss: " + str(rtHammingLoss) + "\n\tPrecision By Label: " + str(rtPrecision) + \
      "\n\tRecall By Label: " + str(rtRecall) + "\n\tLog Loss: " + str(rtLogLoss))

rtBestPipeline = rtModel.bestModel
rtBestModel = rtBestPipeline.stages[-1]

print("numTrees: ", rtBestModel.getOrDefault('numTrees'))
print("maxDepth: ", rtBestModel.getOrDefault('maxDepth'))
print("seed: ", rtBestModel.getOrDefault('seed'))

print(rtModel.validationMetrics)

for i in rtModel.getEstimatorParamMaps():
    print(i.values())

for item, acc in zip(rtModel.getEstimatorParamMaps(), rtModel.validationMetrics):
    print("num_trees is: " + str(item.values()[0]) + " while the max_depth is: " + str(item.values()[2]) + \
          "and the seed is: " + str(item.values()[1]) + " and the accuracy is: " + str(acc))