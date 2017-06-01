import net.sf.javaml.clustering.DensityBasedSpatialClustering
import net.sf.javaml.core.Dataset
import net.sf.javaml.tools.data.FileHandler
import quickml.data.AttributesMap
import quickml.data.instances.RegressionInstance
import quickml.supervised.ensembles.randomForest.randomRegressionForest.RandomRegressionForestBuilder
import quickml.supervised.tree.regressionTree.RegressionTreeBuilder
import java.io.File

/**
 * Created by Евгения on 19.05.2017.
 */

data class OrderPoint(val id: Int, val x: Double, val y: Double, val d: Int, val r: Int, val dd: Int,
                      val s: Int)

data class CarParameters(val number: Int, val capacity: Int)

fun findXYCharacteristics(f: (List<Int>) -> Int): Pair<Int, Int> =
        File("datasets").walkTopDown().toList().filter { it.isFile }.flatMap { file ->
            val lines = file.readLines().drop(9)
            lines.map {
                val (x, y) = it.split("\\s+".toRegex()).drop(2).map(String::toInt)
                x to y
            }
        }.let { pairs ->
            f(pairs.map { (x, _) -> x }) to f(pairs.map { (_, y) -> y })
        }


val maxXAndMaxY = findXYCharacteristics { it.max()!! }
val maxX = maxXAndMaxY.first
val maxY = maxXAndMaxY.second

fun main(args: Array<String>) {
    val (minX, minY) = findXYCharacteristics { it.min()!! }
    val timeByFileName = File("benchmarks.res").readLines().map {                       
        it.split(" ").let { it[0] to it[1].toDouble() }
    }.toMap()
    val filesInGroup = 10
    val trainingSetFilesInGroup = 7
    val trainingSetRegex = Regex("(.*)?_10_[1-$trainingSetFilesInGroup]\\.(.*)")
    val testingSetRegex = Regex("(.*)?_10_[${trainingSetFilesInGroup + 1}|9|(10)](.*)")
    val trainingSetFiles = File("datasets").walkTopDown().toList().filter {
        it.isFile &&
                trainingSetRegex in it.name
    }
    val testingSetFiles = File("datasets").walkTopDown().toList().filter {
        it.isFile &&
                testingSetRegex in it.name
    }
    val trainingSetFeaturesAndTime = mutableListOf<Pair<Map<String, Double>, Double>>()
    val testingSetFeatures = mutableListOf<Pair<Map<String, Double>, String>>()
    for (file in trainingSetFiles) {
        val time = timeByFileName[file.nameWithoutExtension.toLowerCase()]
        if (time != null) {
            val features = extractFeatures(file)

            println("training on ${file.nameWithoutExtension.toLowerCase()}")
            trainingSetFeaturesAndTime.add(Pair(features, time))
        }
    }
    for (file in testingSetFiles) {
        val time = timeByFileName[file.nameWithoutExtension.toLowerCase()]
        if (time != null) {
            val features = extractFeatures(file)
            testingSetFeatures.add(Pair(features, file.name))
        }
    }
    predictWithRandomForest(trainingSetFeaturesAndTime, testingSetFeatures, timeByFileName)
    print("hello")
}

fun extractFeatures(file: File): Map<String, Double>{
    val line = file.readLines()[4]
    val (number, capacity) = line.split("\\s+".toRegex()).drop(1).map(String::toInt)
    val carParameters = CarParameters(number, capacity)
    val datasetParametersNames = listOf("XCOORD.", "YCOORD.", "DEMAND", "READY TIME",
            "DUE DATE", "SERVICE TIME")
    val lines = file.readLines().drop(9)
    val orderPoints = lines.map {
        val (c, x, y, d, r, dd, s) = it.split("\\s+".toRegex()).drop(1).map(String::toInt)
        OrderPoint(c, x.toDouble(), y.toDouble(), d, r, dd, s)
    }
    val stock = orderPoints[0]
    val features = mutableMapOf<String, Double>()

    //Estimations
    features.put("meanDistanceFromXCoordToStock",
            orderPoints.map { Math.abs(it.x - stock.x) }.average())
    features.put("distanceFromMeanXCoordToStock",
            Math.abs(orderPoints.map { it.x }.average() - stock.x))
    features.put("meanDistanceFromYCoordToStock",
            orderPoints.map { Math.abs(it.y - stock.y) }.average())
    features.put("distanceFromMeanYCoordToStock",
            Math.abs(orderPoints.map { it.y }.average() - stock.y))
    features.put("meanMassPerCapacity",
            orderPoints.map { it.d.toDouble() }.average() / carParameters.capacity)
    features.put("meanTimeWindowLength",
            orderPoints.map { it.dd.toDouble() - it.r.toDouble() }.average())

    //Node distribution features
    val distanceMatrix = calculateDistanceMatrixWithNormalizedPoints(orderPoints, maxX, maxY)
    features.put("standardDeviation", calculateStandardDeviation(distanceMatrix.flatten()))
    val (centroidX, centroidY) = calculateCentroid(orderPoints)
    features.put("centroidX", centroidX)
    features.put("centroidY", centroidY)
    features.put("averageDistanceToCentroid",
            calculateAverageDistanceToPoint(orderPoints, Pair(centroidX, centroidY)))
    features.put("fractionOfDistinctDistances",
            calculateFractionOfDistinctDistances(distanceMatrix))
    val (pointsRectangularWidth, pointsRectangularHeight) = calculateRectangularAreaOfPoints(
            orderPoints)
    features.put("pointsRectangularWidth", pointsRectangularWidth)
    features.put("pointsRectangularHeight", pointsRectangularHeight)
    val nNNDs = calculateNormalizedNearestNeighbourDistances(distanceMatrix)
    features.put("nNNDsVariance", calculateVariance(nNNDs))
    features.put("nNNDsCoefficientOfVariation", calculateCoefficientOfVariation(nNNDs))
    features.put("distinctDistancesFraction",
            distanceMatrix.flatten().distinct().size.toDouble() / distanceMatrix.size.square())
    features.put("orderPointsNumber", orderPoints.size.toDouble() - 1)
    val (depotX, depotY) = Pair(orderPoints[0].x, orderPoints[0].y)
    features.put("depotXCoord", depotX)
    features.put("depotYCoord", depotY)
    features.put("distanceFromCentroidToDepot", calculateDistance(Pair(centroidX, centroidY),
            Pair(depotX, depotY)))
    return features
}

fun calculateDbscanClusters(orderPoints: List<OrderPoint>): Array<Dataset>{
    val fileText = orderPoints.map { "${it.id},${it.x},${it.y}" }.joinToString("\n")
    val file = File("dataset")
    file.writeText(fileText)
    val dat = FileHandler.loadDataset(file, 0, ",")
    val clusterer = DensityBasedSpatialClustering()
    return clusterer.cluster(dat)
}

fun predictWithRandomForest(trainingSetFeaturesAndTime: List<Pair<Map<String, Double>, Double>>,
                            testingSetFeatures: List<Pair<Map<String, Double>, String>>,
                            timeByName: Map<String, Double>) {
    val dataset = trainingSetFeaturesAndTime.map {
        RegressionInstance(
                AttributesMap(
                        it.first
                ),
                it.second
        )
    }
    val randomForest = RandomRegressionForestBuilder<RegressionInstance>(
            RegressionTreeBuilder<RegressionInstance>()).buildPredictiveModel(dataset)
    testingSetFeatures.map { println("${randomForest.predict(AttributesMap(it.first))} for" +
            " ${it.second}.") }
}

fun calculateCoefficientOfVariation(listOfNumbers: List<Double>) =
        calculateStandardDeviation(listOfNumbers) / listOfNumbers.average()

fun calculateNormalizedNearestNeighbourDistances(distanceMatrix: List<List<Double>>) =
        distanceMatrix.map { it.filter { it != 0.0 }.min()!! }

fun calculateVariance(listOfNumbers: List<Double>): Double {
    val average = listOfNumbers.average()
    return listOfNumbers.map { (it - average).square() }.average()
}

fun calculateFractionOfDistinctDistances(distanceMatrix: List<List<Double>>) = distanceMatrix.
        flatten().distinctBy{it.toInt()}.size / distanceMatrix.size.square().toDouble()

fun calculateAverageDistanceToPoint(orderPoints: List<OrderPoint>, point: Pair<Double, Double>) =
        orderPoints.map { it.x to it.y }.sumByDouble { calculateDistance(it, point) } /
                orderPoints.size

fun calculateCentroid(orderPoints: List<OrderPoint>) = orderPoints.map { it.x }.average() to
        orderPoints.map { it.y }.average()

fun calculateStandardDeviation(listOfNumbers: List<Double>): Double {
    val average = listOfNumbers.average()
    return Math.sqrt(listOfNumbers.map { (it - average).square() }.average())
}

fun calculateRectangularAreaOfPoints(orderPoints: List<OrderPoint>) = orderPoints.map { it.x }.
        let { it.max()!! - it.min()!!} to orderPoints.map {it.y}.let {it.max()!! - it.min()!!}

fun normalizePoints(orderPoints: List<OrderPoint>, newAreaWidth: Int, newAreaHeight: Int):
        List<OrderPoint>{
    val (pointsAreaWidth, pointsAreaHeight) = calculateRectangularAreaOfPoints(orderPoints)
    return orderPoints.map {it.copy(x = it.x * newAreaWidth/pointsAreaWidth,
            y = it.y * newAreaHeight/pointsAreaHeight)}
}

fun calculateDistanceMatrixWithNormalizedPoints(orderPoints: List<OrderPoint>, newAreaWidth: Int,
                                                newAreaHeight: Int) =
        calculateDistanceMatrix(normalizePoints(orderPoints, newAreaWidth, newAreaHeight))

fun calculateDistanceMatrix(orderPoints: List<OrderPoint>): List<List<Double>> =
        orderPoints.indices.map { i ->
            orderPoints.indices.map { j ->
                calculateDistance(orderPoints[i].x to orderPoints[i].y,
                        orderPoints[j].x to orderPoints[j].y)
            }
        }

fun calculateDistance(p1: Pair<Double, Double>, p2: Pair<Double, Double>): Double {
    val xDistance = p1.first - p2.first
    val yDistance = p1.second - p2.second
    return Math.sqrt((xDistance.square() + yDistance.square()))
}

fun Int.square() = this * this
fun Double.square() = this * this

private operator fun <E> List<E>.component6() = this[5]

private operator fun <E> List<E>.component7() = this[6]