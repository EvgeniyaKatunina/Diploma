
import weka.classifiers.functions.LinearRegression
import weka.classifiers.meta.CVParameterSelection
import weka.core.Attribute
import weka.core.Instances
import java.nio.file.Files

fun createDataForWeka(trainingSetFeaturesAndTime: List<DataItem>): Instances {
    val datasetFile = Files.createTempFile("dataset", "arff").toFile()
    datasetFile.printWriter().use { datasetWriter ->
        datasetWriter.println("@RELATION map")
        val featuresList = trainingSetFeaturesAndTime[0].features.keys
        featuresList.forEach { datasetWriter.println("@ATTRIBUTE $it NUMERIC") }
        datasetWriter.println("@ATTRIBUTE time NUMERIC")
        datasetWriter.println("@DATA")
        trainingSetFeaturesAndTime.forEach { (featureMap, value) ->
            datasetWriter.println((featureMap.values + value).joinToString())
        }
        datasetWriter.close()
    }
    val data = Instances(datasetFile.reader())
    data.setClassIndex(data.numAttributes() - 1)
    return data
}

fun findParametersForLinearRegression(trainingSetFeaturesAndTime: List<DataItem>): CVParameterSelection{
    val data = createDataForWeka(trainingSetFeaturesAndTime)
    val model = weka.classifiers.meta.CVParameterSelection()
    model.buildClassifier(data)
    return model
}

fun trainLinearRegression(trainingSetFeaturesAndTime: List<DataItem>, ridge: Double): LinearRegression {
    val data = createDataForWeka(trainingSetFeaturesAndTime)
    val model = weka.classifiers.functions.LinearRegression()
    model.ridge = ridge
    model.buildClassifier(data)
    return model
}

fun DataItem.toWekaInstanceNoValue() = weka.core.DenseInstance(features.size + 1).apply {
    var i = 0
    for ((k, v) in features + ("answer" to 0.0)) {
        setValue(Attribute(k, i++), v)
    }
}