private val DEFAULT_K_FOLD = 7

fun <Data, Solver, Param : Any> kFoldCrossValidate(
        params: Iterable<Param>,
        data: List<Data>,
        learn: (dataSet: List<Data>, param: Param) -> Solver,
        validate: (solver: Solver, dataSet: List<Data>) -> Double,
        k: Int = DEFAULT_K_FOLD
): Pair<Param, Double> {

    fun itemsInPart(part: Int): Int = when {
        k == 1 -> 0
        part in 0..k - 2 -> data.size / k
        part == k - 1 -> data.size - itemsInPart(0) * (k - 1)
        else -> 0
    }

    class LearnValidateSet(val learnSet: List<Data>, val validateSet: List<Data>)

    val sets = (0..k - 1).map {
        val part1 = ((it - 1) * itemsInPart(0)).coerceAtLeast(0)
        val part2 = itemsInPart(it)
        val part3 = data.size - part1 - part2
        val learningSet = data.take(part1) + data.takeLast(part3)
        val validationSet = data.drop(part1).dropLast(part3)
        LearnValidateSet(learningSet, validationSet)
    }

    return params.map { p ->
        val averageMeasureForP = sets.map {
            val solver = learn(it.learnSet, p)
            validate(solver, it.validateSet)
        }.average()
        p to averageMeasureForP
    }.maxBy { (_, measure) -> measure }!!
}