import numpy as np
import strlearn as sl
from sklearn.base import BaseEstimator, clone

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

from methods.optimization_param_all import OptimizationParamAll


class MooEnsembleAllSVC(BaseEstimator):

    def __init__(self, base_classifier, scale_features=0.5, n_classifiers=10, test_size=0.5, objectives=2, p_size=100):

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.classes = None
        self.test_size = test_size
        self.objectives = objectives
        self.p_size = p_size
        self.ensemble = []
        self.scale_features = scale_features
        self.selected_features = []

    def partial_fit(self, X, y, classes=None):
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        n_features = X.shape[1]

        # Mixed variable problem - genetic operators
        mask = ["real", "real"]
        mask.extend(["binary"] * n_features)
        sampling = MixedVariableSampling(mask, {
            "real": get_sampling("real_random"),
            "binary": get_sampling("bin_random")
        })
        crossover = MixedVariableCrossover(mask, {
            "real": get_crossover("real_sbx"),
            # "real": get_crossover("real_two_point"),
            # sprawdzić różną crossover do real
            "binary": get_crossover("bin_two_point")
        })
        mutation = MixedVariableMutation(mask, {
            "real": get_mutation("real_pm"),
            "binary": get_mutation("bin_bitflip")
        })

        # Create optimization problem
        problem = OptimizationParamAll(X, y, test_size=self.test_size, estimator=self.base_classifier, scale_features=self.scale_features, n_features=n_features, objectives=self.objectives)

        algorithm = NSGA2(
                       pop_size=self.p_size,
                       sampling=sampling,
                       crossover=crossover,
                       mutation=mutation,
                       eliminate_duplicates=True)

        res = minimize(
                       problem,
                       algorithm,
                       ('n_eval', 100),
                       # sprawdź n_gen 100 lub 1000
                       seed=1,
                       verbose=False,
                       save_history=True)

        # F returns all Pareto front solutions in form [-precision, -recall]
        self.solutions = res.F

        # X returns values of hyperparameter C, gamma and binary vector of selected features
        print("X", res.X)
        print("F", self.solutions)
        for result_opt in res.X:
            self.base_classifier = self.base_classifier.set_params(C=result_opt[0], gamma=result_opt[1])
            sf = result_opt[2:].tolist()
            self.selected_features.append(sf)
            # Train new estimator
            candidate = clone(self.base_classifier).fit(X[:, sf], y)
            # Add candidate to the ensemble
            self.ensemble.append(candidate)

        # Pruning based on balanced_accuracy_score
        ensemble_size = len(self.ensemble)
        if ensemble_size > self.n_classifiers:
            bac_array = []
            for clf_id, clf in enumerate(self.ensemble):
                y_pred = clf.predict(X[:, self.selected_features[clf_id]])
                bac = sl.metrics.balanced_accuracy_score(y, y_pred)
                bac_array.append(bac)
            bac_arg_sorted = np.argsort(bac_array)
            self.ensemble_arr = np.array(self.ensemble)
            self.ensemble_arr = self.ensemble_arr[bac_arg_sorted[(len(bac_array)-self.n_classifiers):]]
            self.ensemble = self.ensemble_arr.tolist()

        return self

    def ensemble_support_matrix(self, X):
        # Ensemble support matrix
        return np.array([member_clf.predict_proba(X[:, sf]) for member_clf, sf in zip(self.ensemble, self.selected_features)])

    def predict(self, X):
        # Prediction based on the average support vectors
        ens_sup_matrix = self.ensemble_support_matrix(X)
        average_support = np.mean(ens_sup_matrix, axis=0)
        prediction = np.argmax(average_support, axis=1)
        # Return prediction
        return self.classes_[prediction]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)
