fun <T> rmse(data: List<T>,
             expected: List<Double>,
             estimator: (T) -> Double) =
        Math.sqrt(data.map(estimator).zip(expected) { a, b -> (a - b) * (a - b) }.sum() / data.size)