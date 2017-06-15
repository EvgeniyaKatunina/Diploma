
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import quickml.supervised.ensembles.randomForest.randomRegressionForest.RandomRegressionForest
import weka.attributeSelection.AttributeSelection
import weka.attributeSelection.CorrelationAttributeEval
import weka.attributeSelection.Ranker
import weka.classifiers.functions.LinearRegression
import java.io.File
import java.io.ObjectInputStream
import java.io.ObjectOutputStream

fun serializeDataToBin(data: List<DataItem>, file: File) {
    ObjectOutputStream(file.outputStream()).use { output ->
        output.writeObject(data)
    }
}

fun readDataFromBin(file: File): List<DataItem> =
        ObjectInputStream(file.inputStream()).use { input ->
            @Suppress("UNCHECKED_CAST")
            input.readObject() as List<DataItem>
        }

val validateLinearRmse: (LinearRegression, List<DataItem>) -> Double = { solver, testData ->
    -1 * rmseRelative(testData, testData.map { it.value!! }) { solver.classifyInstance(it.toWekaInstanceNoValue()) }
}

val validateLinearMeanDeviation: (LinearRegression, List<DataItem>) -> Double = { solver, testData ->
    -1 * meanDeviation(testData, testData.map { it.value!! }) { solver.classifyInstance(it.toWekaInstanceNoValue()) }
}

val validateLinearMaxDeviation: (LinearRegression, List<DataItem>) -> Double = { solver, testData ->
    -1 * maxDeviation(testData, testData.map { it.value!! }) { solver.classifyInstance(it.toWekaInstanceNoValue()) }
}

val validateLinearMinDeviation: (LinearRegression, List<DataItem>) -> Double = { solver, testData ->
    -1 * minDeviation(testData, testData.map { it.value!! }) { solver.classifyInstance(it.toWekaInstanceNoValue()) }
}

val linearValidation = mapOf("rmse" to validateLinearRmse, "meanDeviation" to validateLinearMeanDeviation,
        "maxDeviaton" to validateLinearMaxDeviation, "minDeviation" to validateLinearMinDeviation)

fun runLinearRegression(data: List<DataItem>, validate: (LinearRegression, List<DataItem>) -> Double) {
    val trainLinearRegression: (List<DataItem>, Double) -> LinearRegression = { l, ridge -> trainLinearRegression(l, ridge) }

    val ridgeValues = listOf(287.0)

    val (bestRidge, linearRegressionResult) = leaveOneOut(ridgeValues, data, trainLinearRegression, validate)

    println("Best ridge: $bestRidge")
    println("Result: $linearRegressionResult")
}

fun runRandomForest(data: List<DataItem>) {
    val trainRandomForest: (List<DataItem>, Unit) -> RandomRegressionForest = { l, _ -> trainRandomForest(l) }

    val randomForestResult = leaveOneOut(listOf(Unit), data, trainRandomForest, { solver, testData ->
        rmseRelative(testData, testData.map { it.value!! }) { solver.predict(it.toAttributesMap()) }
    })
    println("Random forest RMSE: ${randomForestResult.second}")
}

fun trainNeuralNetwork(allData: List<DataItem>, hiddenNeurons: Int): MultiLayerNetwork {
    val inputNormalizers = allData[0].features.keys.associate { f ->
        f to normalizer(allData.map { it.features[f]!! })
    }
    val outputNormalizer = normalizer(allData.map { it.value!! }.toDoubleArray())
    return trainNeuralNetwork(allData, inputNormalizers, outputNormalizer, hiddenNeurons)
}

fun runNeuralNetwork(data: List<DataItem>) {
    val inputNormalizers = data[0].features.keys.associate { f ->
        f to normalizer(data.map { it.features[f]!! })
    }
    val outputNormalizer = normalizer(data.map { it.value!! }.toDoubleArray())

    val trainNeuralNetwork: (List<DataItem>, Int) -> MultiLayerNetwork =
            { l, neuronsNumber -> trainNeuralNetwork(l, inputNormalizers, outputNormalizer,
                    neuronsNumber) }

    val outMax = data.map { it.value!! }.max()!!
    val outMin = data.map { it.value!! }.min()!!
    fun denormalizeOutput(d: Double) = d * (outMax - outMin) + outMin

    val neuronsNumbers = listOf(1, 2, 3, 4)

    val (bestNeuronsNumber, neuralNetworkResult) = kFoldCrossValidate(neuronsNumbers, data,
            trainNeuralNetwork, { solver, testData ->
        -1 * rmseRelative(testData, testData.map { it.value!! }) {
            denormalizeOutput(solver.output(it.toIndArray(inputNormalizers), false).getDouble(0, 0))
        }
    })
    val normalizedRmse = neuralNetworkResult
    println("Best neurons number: $bestNeuronsNumber")
    println("Neural network RMSE: $normalizedRmse")
}

fun runFeatureSelection(data: List<DataItem>) {
    val dataForWeka = createDataForWeka(data)
    val attsel = AttributeSelection()  // package weka.attributeSelection!
    val eval = CorrelationAttributeEval()
    val search = Ranker()
    //search.setSearchBackwards(true)
    attsel.setEvaluator(eval)
    attsel.setSearch(search)
    attsel.SelectAttributes(dataForWeka)
    attsel.setRanking(true)
    val ranked = attsel.rankedAttributes()
    val indices = attsel.selectedAttributes()
    println(attsel.toResultsString())
    println("hello")
}

fun main(args: Array<String>) {
    println("Reading data")
    val fullData = if (File("data.bin").exists()) readDataFromBin(File("data.bin"))
    else File("datasets").walk()
            .filter { it.isFile }
            .toList()
            .mapNotNull {
                DataItem(extractFeatures(it),
                        (timeByFileName[it.nameWithoutExtension.toLowerCase()] ?: return@mapNotNull null))
            }
            .also { serializeDataToBin(it, File("data.bin")) }
    println("Data read")

    val npointsToData = fullData.groupBy { it.features["orderPointsNumber"] }

//      runFeatureSelection(fullData)

//  runNeuralNetwork(fullData)
//  runRandomForest(fullData)
    for ((k, data) in npointsToData) {
        for ((vName, validate) in linearValidation) {
            println("Running for $k, $vName\n===")
            runLinearRegression(data, validate)
        }
    }

}