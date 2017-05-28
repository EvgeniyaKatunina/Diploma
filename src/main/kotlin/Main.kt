import quickml.data.AttributesMap
import quickml.data.instances.RegressionInstance
import quickml.supervised.ensembles.randomForest.randomRegressionForest.RandomRegressionForestBuilder
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

fun main(args: Array<String>) {
    val (maxX, maxY) = findXYCharacteristics { it.max()!! }
    val (minX, minY) = findXYCharacteristics { it.min()!! }
    val line = File("datasets/homberger_200_customer_instances/C1_2_1.TXT").readLines()[4]
    val (number, capacity) = line.split("\\s+".toRegex()).drop(1).map(String::toInt)
    val carParameters = CarParameters(number, capacity)
    val datasetParametersNames = listOf("XCOORD.", "YCOORD.",
            "DEMAND", "READY TIME", "DUE DATE", "SERVICE TIME")
    val lines = File("datasets/homberger_200_customer_instances/C1_2_1.TXT").readLines().drop(9)
    val orderPoints = lines.map {
        val (c, x, y, d, r, dd, s) = it.split("\\s+".toRegex()).drop(1).map(String::toInt)
        OrderPoint(c, x.toDouble(), y.toDouble(), d, r, dd, s)
    }
    val stock = orderPoints[0]
    val features = mutableListOf<Double>()

    //Estimations
    features.add(orderPoints.map { Math.abs(it.x - stock.x) }.average())
    features.add(Math.abs(orderPoints.map { it.x }.average() - stock.x))
    features.add(orderPoints.map { Math.abs(it.y - stock.y) }.average())
    features.add(Math.abs(orderPoints.map { it.y }.average() - stock.y))
    features.add(orderPoints.map { it.d.toDouble() }.average() / carParameters.capacity)
    features.add(orderPoints.map { it.dd.toDouble() - it.r.toDouble() }.average())

    //Node distribution features
    val distanceMatrix = calculateDistanceMatrixWithNormalizedPoints(orderPoints, maxX, maxY)
    features.add(calculateStandardDeviation(distanceMatrix.flatten()))
    val (centroidX, centroidY) = calculateCentroid(orderPoints)
    features.add(centroidX)
    features.add(centroidY)
    features.add(calculateAverageDistanceToPoint(orderPoints, Pair(centroidX, centroidY)))
    features.add(calculateFractionOfDistinctDistances(distanceMatrix))
    val (pointsRectangularWidth, pointsRectangularHeight) = calculateRectangularAreaOfPoints(
            orderPoints)
    features.add(pointsRectangularWidth)
    features.add(pointsRectangularHeight)
    val nNNDs = calculateNormalizedNearestNeighbourDistances(distanceMatrix)
    features.add(calculateVariance(nNNDs))
    features.add(calculateCoefficientOfVariation(nNNDs))
    /*val fileText = orderPoints.map {"${it.id} ${it.x} ${it.y}"}.joinToString("\n")
    val file = File("dataset")
    file.writeText(fileText)
    val dat = FileHandler.loadDataset(file, 0, "\n")
    val clusterer = DensityBasedSpatialClustering(10000000.0, 4)
    val datasets = clusterer.cluster(dat)
    val file = File("iris.data")
    file.writeText(fileText)
    val dat = FileHandler.loadDataset(file, 0, "\n")
    val clusterer = DensityBasedSpatialClustering()
    val datasets = clusterer.cluster(dat)*/
    predictWithRandomForest(datasetParametersNames, orderPoints)
    print("hello")
}

fun predictWithRandomForest(parametersNames: List<String>, orderPoints: List<OrderPoint>) {
    val dataset = orderPoints.map {
        RegressionInstance(
                AttributesMap(
                        parametersNames.zip(listOf(it.x.toInt(), it.y.toInt(), it.d, it.r, it.dd, it.s)).toMap()
                ),
                0.34
        )
    }
    val randomForest = RandomRegressionForestBuilder<RegressionInstance>().buildPredictiveModel(dataset)
    //randomForest.predict()
}

fun calculateCoefficientOfVariation(listOfNumbers: List<Double>) =
        calculateStandardDeviation(listOfNumbers) / listOfNumbers.average()

fun calculateNormalizedNearestNeighbourDistances(distanceMatrix: List<List<Double>>) =
        distanceMatrix.map { it.minBy { it > 0 }!! }

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