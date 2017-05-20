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

fun main(args: Array<String>) {
    val (maxX, maxY) = findMaxXY()
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
    //

    print("hello")
}

private operator fun <E> List<E>.component6() = this[5]

private operator fun <E> List<E>.component7() = this[6]
