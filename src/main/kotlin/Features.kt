import java.io.File

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

private operator fun <E> List<E>.component6() = this[5]
private operator fun <E> List<E>.component7() = this[6]