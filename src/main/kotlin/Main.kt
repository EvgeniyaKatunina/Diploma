import java.io.File

/**
 * Created by Евгения on 19.05.2017.
 */

data class OrderPoint(val id: Int, val x: Int, val y: Int, val d: Int, val r: Int, val dd: Int,
                      val s: Int)

fun main(args: Array<String>) {
    val lines = File("datasets/homberger_200_customer_instances/C1_2_1.TXT").readLines().drop(9)
    val orderPoints = lines.map {
        val (c, x, y, d, r, dd, s) = it.split("\\s+".toRegex()).drop(1).map(String::toInt)
        OrderPoint(c, x, y, d, r, dd, s)
    }
}

private operator fun <E> List<E>.component6() = this[5]

private operator fun <E> List<E>.component7() = this[6]
