import java.io.File

/**
 * Created by Евгения on 19.05.2017.
 */

data class OrderPoint(val id: Int, val x: Int, val y: Int, val d: Int, val r: Int, val dd: Int,
                      val s: Int)

data class CarParameters(val number: Int, val capacity: Int)

fun findMaxXY(): Pair<Int, Int> =
        File("datasets").walkTopDown().toList().filter { it.isFile }.flatMap { file ->
            val lines = file.readLines().drop(9)
            lines.map {
                val (x, y) = it.split("\\s+".toRegex()).drop(2).map(String::toInt)
                x to y
            }
        }.let { pairs ->
            pairs.map { (x, _) -> x }.max()!! to pairs.map { (_, y) -> y }.max()!!
        }

fun findMinXY(): Pair<Int, Int> =
        File("datasets").walkTopDown().toList().filter { it.isFile }.flatMap { file ->
            val lines = file.readLines().drop(9)
            lines.map {
                val (x, y) = it.split("\\s+".toRegex()).drop(2).map(String::toInt)
                x to y
            }
        }.let { pairs ->
            pairs.map { (x, _) -> x }.min()!! to pairs.map { (_, y) -> y }.min()!!
        }

fun main(args: Array<String>) {
    val (maxX, maxY) = findMaxXY()
    val (minX, minY) = findMinXY()
    val line = File("datasets/homberger_200_customer_instances/C1_2_1.TXT").readLines()[4]
    val (number, capacity) = line.split("\\s+".toRegex()).drop(1).map(String::toInt)
    val carParameters = CarParameters(number, capacity)
    val lines = File("datasets/homberger_200_customer_instances/C1_2_1.TXT").readLines().drop(9)
    val orderPoints = lines.map {
        val (c, x, y, d, r, dd, s) = it.split("\\s+".toRegex()).drop(1).map(String::toInt)
        OrderPoint(c, x, y, d, r, dd, s)
    }
    val stock = orderPoints[0]
    val features = mutableListOf<Double>()
    //Estimations
    features.add(orderPoints.map { Math.abs(it.x.toDouble() - stock.x) }.average())
    features.add(Math.abs(orderPoints.map { it.x.toDouble() }.average() - stock.x))
    features.add(orderPoints.map { Math.abs(it.y.toDouble() - stock.y) }.average())
    features.add(Math.abs(orderPoints.map { it.y.toDouble() }.average() - stock.y))
    features.add(orderPoints.map { it.d.toDouble() }.average() / carParameters.capacity)
    features.add(orderPoints.map { it.dd.toDouble() - it.r.toDouble() }.average())
    //Node distribution features
    val distanceMatrix = calculateDistanceMatrixWithNormalizedPoints(orderPoints, maxX, maxY)

    print("hello")
}

fun normalizePoints(orderPoints: List<OrderPoint>, newAreaWidth: Int, newAreaHeight: Int):
        List<OrderPoint>{
    val pointsAreaWidth = orderPoints.map { it.x }.max()!! - orderPoints.map { it.x }.min()!!
    val pointsAreaHeight = orderPoints.map { it.y }.max()!! - orderPoints.map { it.y }.min()!!
    return orderPoints.map { OrderPoint(it.id, it.x * newAreaWidth/pointsAreaWidth,
            it.y * newAreaHeight/pointsAreaHeight, it.d, it.r, it.dd, it.s) }
}

fun calculateDistanceMatrixWithNormalizedPoints(orderPoints: List<OrderPoint>, newAreaWidth: Int,
                                                newAreaHeight: Int) =
        calculateDistanceMatrix(normalizePoints(orderPoints, newAreaWidth, newAreaHeight))

fun calculateDistanceMatrix(orderPoints: List<OrderPoint>) : List<List<Double>>{
    val distanceMatrix = mutableListOf<MutableList<Double>>()
    for (i in 0..orderPoints.lastIndex){
        distanceMatrix.add(mutableListOf<Double>())
    }
    for (i in 0..orderPoints.lastIndex){
        for (j in 0..orderPoints.lastIndex){
            distanceMatrix[i].add(calculateDistance(orderPoints[i].x to orderPoints[i].y,
                    orderPoints[j].x to orderPoints[j].y))
        }
    }
    return distanceMatrix
}

fun calculateDistance(p1: Pair<Int, Int>, p2: Pair<Int, Int>) : Double{
    val xDistance = p1.first - p2.first
    val yDistance = p1.second - p2.second
    return Math.sqrt((xDistance * xDistance + yDistance * yDistance).toDouble())
}

private operator fun <E> List<E>.component6() = this[5]

private operator fun <E> List<E>.component7() = this[6]
