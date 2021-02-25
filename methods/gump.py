"""Forrest Gump."""
import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode
from torch import cdist, from_numpy
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt


class GUMP(ClassifierMixin, BaseEnsemble):
    """
    Bootstrapped Decision Forest.
    """
    def __init__(self, base_estimator=DecisionTreeClassifier(random_state=1410), n_estimators=50, p=2, random_state=None, decision=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.p = p
        self.random_state = random_state
        self.decision = decision

    def fit(self, X, y):
        self.X, self.y = check_X_y(X, y)
        self.ensemble_ = []
        self.roots = []
        self.classes_, _ = np.unique(self.y, return_inverse=True)
        np.random.seed(self.random_state)

        # Distances n_samples x n_samples
        self.distances = cdist(from_numpy(self.X), from_numpy(self.X), p=self.p).numpy()
        # All samples
        indxs = np.array(list(range(self.X.shape[0])))

        plt.scatter(X[:,0], X[:,1], c="black")

        for base_clf in range(self.n_estimators):
            if base_clf == 0:
                root = np.random.randint(0, self.X.shape[0]-1)
                self.roots.append(root)
                n1 = self.distances[root]/np.sum(self.distances[root])
                n2 = np.max(self.distances[root]) - self.distances[root]
                n2 = n2/np.sum(n2)
                bs_indx = np.random.choice(indxs, size=int(self.X.shape[0]), replace=True, p=n2)
                bs_X, bs_y = self.X[bs_indx], self.y[bs_indx]
                self.ensemble_.append(clone(self.base_estimator).fit(bs_X, bs_y))
                # plt.scatter(X[bs_indx,0], X[bs_indx,1], c="green")
                # plt.scatter(X[root,0], X[root,1], c="red")

            else:
                bs_dist = np.mean(self.distances[self.roots], axis=0)
                max_dist = np.argmax(bs_dist)
                self.roots.append(max_dist)

                n2 = np.max(self.distances[max_dist]) - self.distances[max_dist]
                n2 = n2/np.sum(n2)
                bs_indx = np.random.choice(indxs, size=int(self.X.shape[0]), replace=True, p=n2)
                bs_X, bs_y = self.X[bs_indx], self.y[bs_indx]
                self.ensemble_.append(clone(self.base_estimator).fit(bs_X, bs_y))
                # plt.scatter(X[bs_indx,0], X[bs_indx,1], c="green")
                # plt.scatter(X[max_dist,0], X[max_dist,1], c="red")
                # plt.tight_layout()
                # plt.savefig("foo.png")
                # exit()
        # print(len(self.ensemble_))
        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X.shape[1]:
            raise ValueError("number of features does not match")

        if self.decision == "SACC":
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)
        elif self.decision == "MV":
            predictions = np.array([member_clf.predict(X) for member_clf in self.ensemble_])
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        elif self.decision == "S":
            predictions = np.array([member_clf.predict(X) for member_clf in self.ensemble_]).T
            # print(predictions)
            distances = np.argsort(cdist(from_numpy(X), from_numpy(self.X[self.roots]), p=self.p).numpy(), axis=1)[:,0]
            # print(distances)
            prediction = []
            for sample in range(X.shape[0]):
                prediction.append(predictions[sample][distances[sample]])
            prediction = np.array(prediction)
            # print(preds)
            # exit()

        # Return prediction
        return prediction
