
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*

fun normalizer(doubles: Iterable<Double>) = normalizer(doubles.min()!!, doubles.max()!!)
fun normalizer(array: DoubleArray) = normalizer(array.min()!!, array.max()!!)
fun normalizer(min: Double, max: Double): (Double) -> Double = { (it - min) / (max - min) }

fun trainNeuralNetwork(trainData: List<DataItem>,
                       inputNormalizers: Map<String, (Double) -> Double>,
                       outputNormalizer: (Double) -> Double
                       ): MultiLayerNetwork {
    val nFeatures = trainData[0].features.size

    val n = trainData.size
    val featureArrays = trainData[0].features.keys.map { feature ->
        val transform = inputNormalizers[feature]!!
        Nd4j.create(trainData.map { item -> transform(item.features[feature]!!) }.toDoubleArray(), intArrayOf(n, 1))
    }
    val inputData = Nd4j.hstack(featureArrays)

    val output = Nd4j.create(trainData.map { outputNormalizer(it.value!!) }.toDoubleArray(), intArrayOf(n, 1))
    val dataIterator = ListDataSetIterator(org.nd4j.linalg.dataset.DataSet(inputData, output).asList().apply(Collections::shuffle))

    val epochs = 600
    val seed = 12345
    val learningRate = 0.01

    val numHiddenNodes = 10
    val config = NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(1)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list()
            .layer(0, DenseLayer.Builder().nIn(nFeatures).nOut(numHiddenNodes)
                    .activation(Activation.IDENTITY).build())
            .layer(1, DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                    .activation(Activation.IDENTITY).build())
            .layer(2, OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .activation(Activation.IDENTITY)
                    .nIn(numHiddenNodes).nOut(1).build())
            .pretrain(false).backprop(true).build()

    val net = MultiLayerNetwork(config)
    net.init()

    for (i in 0..epochs - 1) {
        if (i % 100 == 0) println("Training: $i")
        dataIterator.reset()
        net.fit(dataIterator)
    }

    return net
}

fun DataItem.toIndArray(normalizers: Map<String, (Double) -> Double>) =
        Nd4j.create(features.mapValues { (k, v) -> normalizers[k]!!(v) }.values.toDoubleArray(), intArrayOf(1, features.size))