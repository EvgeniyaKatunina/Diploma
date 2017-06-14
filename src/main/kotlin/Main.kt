
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import quickml.supervised.ensembles.randomForest.randomRegressionForest.RandomRegressionForest
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

fun runLinearRegression(data: List<DataItem>) {
    val trainLinearRegression: (List<DataItem>, Double) -> LinearRegression = { l, ridge ->
        val result = trainLinearRegression(l, ridge)
        println(result)
        result
    }

    val ridgeValues = listOf(287.0)

    val (bestRidge, linearRegressionResult) = leaveOneOut(ridgeValues, data, trainLinearRegression, { solver, testData ->
        -1 * rmseRelative(testData, testData.map { it.value!! }) { solver.classifyInstance(it.toWekaInstanceNoValue()) }
    })

    println("Best ridge: $bestRidge")
    println("Linear regression RMSE: $linearRegressionResult")
}

fun runRandomForest(data: List<DataItem>) {
    val trainRandomForest: (List<DataItem>, Unit) -> RandomRegressionForest = { l, _ -> trainRandomForest(l) }

    val randomForestResult = leaveOneOut(listOf(Unit), data, trainRandomForest, { solver, testData ->
        rmseRelative(testData, testData.map { it.value!! }) { solver.predict(it.toAttributesMap()) }
    })
    println("Random forest RMSE: ${randomForestResult.second}")
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

    val neuronsNumbers = (0..5).map {5 + it * 2}

    val (bestNeuronsNumber ,neuralNetworkResult) = kFoldCrossValidate(neuronsNumbers, data, trainNeuralNetwork, { solver, testData ->
        rmseRelative(testData, testData.map { it.value!! }) {
            denormalizeOutput(solver.output(it.toIndArray(inputNormalizers), false).getDouble(0, 0))
        }
    })
    val normalizedRmse = neuralNetworkResult
    println("Best neurons number: $bestNeuronsNumber")
    println("Neural network RMSE: $normalizedRmse")
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
    

    runNeuralNetwork(fullData)
//    runRandomForest(fullData)
//    runLinearRegression(fullData)
}