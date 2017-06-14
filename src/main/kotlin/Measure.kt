fun <T> rmseRelative(data: List<T>,
                     expected: List<Double>,
                     estimator: (T) -> Double) =
        Math.sqrt(data.map(estimator).zip(expected) { a, b ->
            (a - b) / b * (a - b) / b
        }.sum() / data.size)

fun <T> maxDeviation(data: List<T>,
                     expected: List<Double>,
                     estimator: (T) -> Double) =
        data.map(estimator).zip(expected) { a, b ->
            Math.abs(a - b)
        }.max()

fun <T> minDeviation(data: List<T>,
                     expected: List<Double>,
                     estimator: (T) -> Double) =
        data.map(estimator).zip(expected) { a, b ->
            Math.abs(a - b)
        }.min()

fun <T> meanDeviation(data: List<T>,
                     expected: List<Double>,
                     estimator: (T) -> Double) =
        data.map(estimator).zip(expected) { a, b ->
            Math.abs(a - b)
        }.average()