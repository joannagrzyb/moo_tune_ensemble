from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
import math
import warnings
from sklearn.base import clone

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

from methods.optimization_param import OptimizationParam


class MooEnsembleSVC(BaseEstimator):

    def __init__(self, base_classifier, scale_features=0.5, number_of_classifiers=10, test_size=0.5, pareto_decision='all', objectives=2, p_size=100):

        self.base_classifier = base_classifier
        self.number_of_classifiers = number_of_classifiers
        self.classifier_array = []
        self.classifier_weights = []
        self.minority_name = None
        self.majority_name = None
        self.classes = None
        self.label_encoder = None
        self.iterator = 1
        self.test_size = test_size
        self.objectives = objectives
        self.p_size = p_size
        self.hyperparameters = None
        self.pareto_decision = pareto_decision
        self.ensemble = []
        self.scale_features = scale_features

    def partial_fit(self, X, y, classes=None):
        n_features = X.shape[1]

        # Mixed variable problem optimization
        mask = ["real", "real"]
        mask.extend(["binary"] * n_features)
        sampling = MixedVariableSampling(mask, {
            "real": get_sampling("real_random"),
            "binary": get_sampling("bin_random")
        })
        crossover = MixedVariableCrossover(mask, {
            "real": get_crossover("real_two_point"),
            "binary": get_crossover("bin_two_point")
        })
        mutation = MixedVariableMutation(mask, {
            "real": get_mutation("real_pm"),
            "binary": get_mutation("bin_bitflip")
        })

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        problem = OptimizationParam(X, y, test_size=self.test_size, estimator=self.base_classifier, scale_features=self.scale_features, n_features=n_features, objectives=self.objectives)

        algorithm = NSGA2(
                       pop_size=self.p_size,
                       sampling=sampling,
                       crossover=crossover,
                       mutation=mutation,
                       eliminate_duplicates=True)

        res = minimize(
                       problem,
                       algorithm,
                       ('n_eval', 1000),
                       seed=1,
                       verbose=False,
                       save_history=True)

        print(res.algorithm)
        print(res.G)
        print(res.CV)
        print(res.pf)

        # Select solution from the Pareto front
        # F returns all solutions in form [-precision, -recall]
        self.solutions = res.F
        print("X", res.X)
        print("F", self.solutions)

        # X returns values of hyperparameter C, gamma and binary vector of selected features
        if self.pareto_decision == 'precision':
            index = np.argmin(self.solutions[:, 0], axis=0)
            self.hyperparameters = res.X[index]
            # create_ensemble(self)
        elif self.pareto_decision == 'recall':
            index = np.argmin(self.solutions[:, 1], axis=0)
            self.hyperparameters = res.X[index]
            # create_ensemble(self)
        elif self.pareto_decision == 'all':
            for i in res.X:
                self.hyperparameters = i
                self.base_classifier = self.base_classifier.set_params(C=self.hyperparameters[0], gamma=self.hyperparameters[1])
                # Train new estimator
                self.candidate = self.base_classifier.fit(X, y)
                # Add candidate to the ensemble
                self.ensemble.append(self.candidate)

        # Pruning ?

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble])

    def predict(self, X):
        # Check is fit had been called
        # check_is_fitted(self, "classes_")
        # X = check_array(X)
        # if X.shape[1] != self.X_.shape[1]:
        #     raise ValueError("number of features does not match")

        # predykcja na podstawie średnich wektorów wsparć
        ens_sup_matrix = self.ensemble_support_matrix(X)
        average_support = np.mean(ens_sup_matrix, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)
        # return np.average(probas_, axis=0, weights=self.classifier_weights)
