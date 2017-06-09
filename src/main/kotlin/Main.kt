
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
    val trainLinearRegression: (List<DataItem>, Unit) -> LinearRegression = { l, _ ->
        val result = trainLinearRegression(l)
        println(result)
        result
    }

    val linearRegressionResult = kFoldCrossValidate(listOf(Unit), data, trainLinearRegression, { solver, testData ->
        rmse(testData, testData.map { it.value!! }) { solver.classifyInstance(it.toWekaInstanceNoValue()) }
    }).second
    println("Linear regression RMSE: $linearRegressionResult")
}

fun runRandomForest(data: List<DataItem>) {
    val trainRandomForest: (List<DataItem>, Unit) -> RandomRegressionForest = { l, _ -> trainRandomForest(l) }

    val randomForestResult = kFoldCrossValidate(listOf(Unit), data, trainRandomForest, { solver, testData ->
        rmse(testData, testData.map { it.value!! }) { solver.predict(it.toAttributesMap()) }
    })
    println("Random forest RMSE: ${randomForestResult.second}")
}

fun runNeuralNetwork(data: List<DataItem>) {
    val inputNormalizers = data[0].features.keys.associate { f -> f to normalizer(data.map { it.features[f]!! }) }
    val outputNormalizer = normalizer(data.map { it.value!! }.toDoubleArray())
    val trainNeuralNetwork: (List<DataItem>, Unit) -> MultiLayerNetwork = { l, _ -> trainNeuralNetwork(l, inputNormalizers, outputNormalizer) }

    val neuralNetworkResult = kFoldCrossValidate(listOf(Unit), data, trainNeuralNetwork, { solver, testData ->
        rmse(testData, testData.map { outputNormalizer(it.value!!) }) { solver.output(it.toIndArray(inputNormalizers), false).getDouble(0, 0) }
    })
    val normalizedRmse = neuralNetworkResult.second
    val outMax = data.map { it.value!! }.max()!!
    val outMin = data.map { it.value!! }.min()!!
    println("Neural network RMSE: ${normalizedRmse * (outMax - outMin) + outMin}")
}

fun main(args: Array<String>) {
    println("Reading data")
    val fullData = if (File("data.bin").exists()) readDataFromBin(File("data.bin")) else File("datasets").walk()
            .filter { it.isFile }
            .toList()
            .mapNotNull { DataItem(extractFeatures(it), (timeByFileName[it.nameWithoutExtension.toLowerCase()] ?: return@mapNotNull null)) }
            .also { serializeDataToBin(it, File("data.bin")) }
    println("Data read")

    runNeuralNetwork(fullData)
    runLinearRegression(fullData)
    runRandomForest(fullData)
}