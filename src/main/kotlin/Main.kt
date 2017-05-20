import java.io.File

/**
 * Created by Евгения on 19.05.2017.
 */

data class OrderPoint(val id: Int, val x: Int, val y: Int, val d: Int, val r: Int, val dd: Int,
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
    val pointsAreaWidth = orderPoints.map { it.x }.let { it.max()!! - it.min()!!}
    val pointsAreaHeight = orderPoints.map {it.y}.let {it.max()!! - it.min()!!}
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

fun calculateDistance(p1: Pair<Int, Int>, p2: Pair<Int, Int>): Double {
    val xDistance = p1.first - p2.first
    val yDistance = p1.second - p2.second
    return Math.sqrt((xDistance.square() + yDistance.square()).toDouble())
}

fun Int.square() = this * this

private operator fun <E> List<E>.component6() = this[5]

private operator fun <E> List<E>.component7() = this[6]