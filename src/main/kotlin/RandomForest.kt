import quickml.data.AttributesMap
import quickml.data.instances.RegressionInstance
import quickml.supervised.ensembles.randomForest.randomRegressionForest.RandomRegressionForest
import quickml.supervised.ensembles.randomForest.randomRegressionForest.RandomRegressionForestBuilder
import quickml.supervised.tree.regressionTree.RegressionTreeBuilder

fun trainRandomForest(trainingSet: List<DataItem>): RandomRegressionForest {
    val dataset = trainingSet.map {
        RegressionInstance(
                AttributesMap(it.features),
                it.value
        )
    }
    val randomForest = RandomRegressionForestBuilder<RegressionInstance>(
            RegressionTreeBuilder<RegressionInstance>()).buildPredictiveModel(dataset)
    return randomForest
}

fun DataItem.toAttributesMap() = AttributesMap(features)