import numpy as np
import autograd.numpy as anp
from pymoo.model.problem import Problem
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import strlearn as sl


class OptimizationParam(Problem):
    def __init__(self, X, y, test_size, estimator, scale_features, n_features, n_param=2, objectives=2, random_state=0, feature_names=None):

        self.estimator = estimator
        self.test_size = test_size
        self.objectives = objectives
        self.n_param = n_param
        self.scale_features = scale_features
        self.n_features = n_features

        # to chyba głupie, ale trochę potrzebne do określenia ilości wybranych cech
        self.feature_names = feature_names
        if self.feature_names == None:
            self.feature_names = [i for i in range(self.n_features)]
        self.n_max = len(self.feature_names)

        # If test size is not specify or it is 0, everything is took to test and train
        if self.test_size != 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = np.copy(X), np.copy(y), np.copy(X), np.copy(y)

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        xl = 1E-3 * anp.ones(2)
        xu = 1E4 * anp.ones(2)
        n_variable = self.n_param + self.n_features

        super().__init__(n_var=n_variable, n_obj=objectives,
                         n_constr=1, elementwise_evaluation=True, xl=xl, xu=xu)

    def validation(self, x):
        # x: a two dimensional matrix where each row is a point to evaluate and each column a variable
        C = x[0]
        gamma = x[1]
        clf = clone(self.estimator.set_params(C=C, gamma=gamma))
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        metrics = [sl.metrics.precision(self.y_test, y_pred), sl.metrics.recall(self.y_test, y_pred)]
        return metrics

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)
        # Function F is always minimize, but the minus sign (-) before F means maximize
        f1 = -1 * (scores[0])   # Precision
        f2 = -1 * (scores[1])   # Recall
        out["F"] = anp.column_stack(np.array([f1, f2]))

        # Function constraint to select specific numbers of features:
        number = int((1 - self.scale_features) * self.n_max)
        out["G"] = (self.n_max - np.sum(x) - number) ** 2
