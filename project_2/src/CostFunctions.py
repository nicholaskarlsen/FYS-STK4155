
class CostFunction:
    def __init__(self):
        pass

    def __call__(self, x, y, predictors, *args):
        pass

    def gradient(self, x, y, predictors, *args):
        pass


class OLS(CostFunction):
    def __init__(self):
        pass

    def __call__(self, X, y, predictors):
        return (y - X @ predictors).T @ (y - X @ predictors)

    def gradient(self, X, y, predictors):
        pass