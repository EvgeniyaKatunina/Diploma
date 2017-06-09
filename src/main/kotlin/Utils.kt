fun Int.square() = this * this
fun Double.square() = this * this

fun calculateDistance(p1: Pair<Double, Double>, p2: Pair<Double, Double>): Double {
    val xDistance = p1.first - p2.first
    val yDistance = p1.second - p2.second
    return Math.sqrt((xDistance.square() + yDistance.square()))
}

fun calculateDistanceMatrix(orderPoints: List<OrderPoint>): List<List<Double>> =
        orderPoints.indices.map { i ->
            orderPoints.indices.map { j ->
                calculateDistance(orderPoints[i].x to orderPoints[i].y,
                        orderPoints[j].x to orderPoints[j].y)
            }
        }