import numpy as np
import strlearn as sl
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from torch import cdist, from_numpy
import matplotlib.pyplot as plt

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

from methods.optimization_param import OptimizationParam
from utils.diversity import calc_diversity_measures, calc_diversity_measures2


class MooEnsembleSVC(BaseEstimator):

    def __init__(self, base_classifier, scale_features=0.5, n_classifiers=10, test_size=0.5, objectives=2, p_size=100, predict_decision="ASV", p_minkowski=2):

        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.classes = None
        self.test_size = test_size
        self.objectives = objectives
        self.p_size = p_size
        self.scale_features = scale_features
        self.selected_features = []
        self.predict_decision = predict_decision
        self.p_minkowski = p_minkowski

    def partial_fit(self, X, y, classes=None):
        self.X, self.y = X, y
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(self.y, return_inverse=True)

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
                       # sprawdź n_gen 100 lub 1000
                       seed=1,
                       verbose=False,
                       save_history=True)

        # F returns all Pareto front solutions in form [-precision, -recall]
        self.solutions = res.F

        # X returns values of hyperparameter C, gamma and binary vector of selected features
        # print("X", res.X)
        # print("F", self.solutions)
        # for result_opt in res.X:
        #     self.base_classifier = self.base_classifier.set_params(C=result_opt[0], gamma=result_opt[1])
        #     sf = result_opt[2:].tolist()
        #     self.selected_features.append(sf)
        #     # Train new estimator
        #     candidate = clone(self.base_classifier).fit(X[:, sf], y)
        #     # Add candidate to the ensemble
        #     self.ensemble.append(candidate)

        # """
        # Bootstraping - GUMP
        print("BOOTSTRAPPING")
        self.roots = []
        # Distances n_samples x n_samples
        # p - parameter Minkowski distance, if p=2 Euclidean distance, if p=1 Manhattan distance, if 0<p<1 it's better for more dimenesions
        self.distances = cdist(from_numpy(self.X), from_numpy(self.X), p=self.p_minkowski).numpy()
        # All samples
        indxs = np.array(list(range(self.X.shape[0])))

        plt.scatter(X[:, 0], X[:, 1], c="black")
        plt.savefig("scatter.png", bbox_inches='tight')

        for result_opt in res.X:
            self.base_classifier = self.base_classifier.set_params(C=result_opt[0], gamma=result_opt[1])
            sf = result_opt[2:].tolist()
            self.selected_features.append(sf)
            for clf_id in range(self.n_classifiers):
                if clf_id == 0:
                    # dla pierwszego modelu w ensemble inaczej wybiera się próbki
                    root = np.random.randint(0, self.X.shape[0]-1)
                    self.roots.append(root)
                    n2 = np.max(self.distances[root]) - self.distances[root]
                    n2 = n2/np.sum(n2)
                    # bs_indx zwraca indeksy wybranych próbek
                    bs_indx = np.random.choice(indxs, size=int(self.X.shape[0]), replace=True, p=n2)
                    bs_X = self.X[bs_indx, :]
                    bs_X = bs_X[:, sf]
                    bs_y = self.y[bs_indx]

                    # print("KLASY:", np.unique(self.y[bs_indx]))
                    # print("maj", sum(1 for i in self.y[bs_indx] if i == 0))
                    # print("min", sum(1 for i in self.y[bs_indx] if i == 1))

                    # Jeśli po Bootstrapingu jest tylko jedna klasa (a może się tak zdarzyć przy danych niezbalansowanych, to wtedy należy usunąć n-próbek (1 lub 2 lub dać to jako parametr), a następnie dodać n-próbek losowych z klasy mniejszościowej) - kod nieskończony
                    if len(np.unique(self.y[bs_indx])) == 1:
                        print("KLASY:", np.unique(self.y[bs_indx]))
                        # delete 2 last elements
                        bs_X.pop(bs_indx[-1])
                        bs_y.pop(bs_indx[-1])
                        bs_X.pop(bs_indx[-2])
                        bs_y.pop(bs_indx[-2])
                        # add 2 elements of different (minority) class
                        # if self.y == 1:
                        # indx_min = self.y.index(1)
                        # bs_X.append(self.X)
                    # klasyfikator uczymy na wybranych danych (bs_X i bs_y), a następnie dodajemy go do ensemble
                    candidate = clone(self.base_classifier).fit(bs_X, bs_y)
                    self.ensemble.append(candidate)
                    plt.scatter(X[bs_indx, 0], X[bs_indx, 1], c="green")
                    plt.scatter(X[root, 0], X[root, 1], c="red")

                else:
                    bs_dist = np.mean(self.distances[self.roots], axis=0)
                    max_dist = np.argmax(bs_dist)
                    self.roots.append(max_dist)
                    n2 = np.max(self.distances[max_dist]) - self.distances[max_dist]
                    n2 = n2/np.sum(n2)
                    bs_indx = np.random.choice(indxs, size=int(self.X.shape[0]), replace=True, p=n2)
                    bs_X = self.X[bs_indx, :]
                    bs_X = bs_X[:, sf]
                    bs_y = self.y[bs_indx]
                    candidate = clone(self.base_classifier).fit(bs_X, bs_y)
                    self.ensemble.append(candidate)
                    plt.scatter(X[bs_indx, 0], X[bs_indx, 1], c="green")
                    plt.scatter(X[max_dist, 0], X[max_dist, 1], c="red")
                    plt.tight_layout()
                    plt.savefig("scatter_end.png")
                    plt.close()
        # """

        """
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
        """

        return self

    def fit(self, X, y, classes=None):
        self.ensemble = []
        self.partial_fit(X, y, classes)

    def ensemble_support_matrix(self, X):
        # Ensemble support matrix
        return np.array([member_clf.predict_proba(X[:, sf]) for member_clf, sf in zip(self.ensemble, self.selected_features)])

    def predict(self, X):
        # Prediction based on the Average Support Vectors - to wybierz!
        if self.predict_decision == "ASV":
            ens_sup_matrix = self.ensemble_support_matrix(X)
            average_support = np.mean(ens_sup_matrix, axis=0)
            # print("AVG:", average_support)
            prediction = np.argmax(average_support, axis=1)
        # Prediction based on the Majority Voting
        elif self.predict_decision == "MV":
            predictions = np.array([member_clf.predict(X) for member_clf in self.ensemble_])
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        # return prediction

        return self.classes_[prediction]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)

    def calculate_diversity(self):
        if len(self.ensemble) > 1:
            # All measures for whole ensemble
            self.entropy_measure_e, self.k0, self.kw, self.disagreement_measure, self.q_statistic_mean = calc_diversity_measures(self.X, self.y, self.ensemble, self.selected_features, p=0.01)
            # entropy_measure_e: E varies between 0 and 1, where 0 indicates no difference and 1 indicates the highest possible diversity.
            # kw - Kohavi-Wolpert variance
            # Q-statistic: <-1, 1>
            # Q = 0 statistically independent classifiers
            # Q < 0 classifiers commit errors on different objects
            # Q > 0 classifiers recognize the same objects correctly

            return(self.entropy_measure_e, self.kw, self.disagreement_measure, self.q_statistic_mean)

            """
            # k - measurement of interrater agreement
            self.kkk = []
            for sf in self.selected_features:
                # Calculate mean accuracy on training set
                p = np.mean(np.array([accuracy_score(self.y, member_clf.predict(self.X[:, sf])) for clf_id, member_clf in enumerate(self.ensemble)]))
                self.k = calc_diversity_measures2(self.X, self.y, self.ensemble, self.selected_features, p, measure="k")
                self.kkk.append(self.k)
            return self.kkk
            """
