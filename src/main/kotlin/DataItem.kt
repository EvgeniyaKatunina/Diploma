import java.io.File
import java.io.Serializable

data class OrderPoint(val id: Int, val x: Double, val y: Double, val d: Int, val r: Int, val dd: Int, val s: Int)

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

val timeByFileName by lazy {
    File("benchmarks.res").readLines().map {
        it.split(" ").let { it[0] to it[1].toDouble() }
    }.toMap()
}

data class DataItem(val features: Map<String, Double>, val value: Double? = null) : Serializable