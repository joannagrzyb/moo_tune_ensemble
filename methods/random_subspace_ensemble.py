from sklearn.base import BaseEstimator, clone
import numpy as np


class RandomSubspaceEnsemble(BaseEstimator):
    def __init__(self, base_classifier, n_members=100, subspace_size=3):
        self.base_classifier = base_classifier
        self.n_members = n_members
        self.subspace_size = subspace_size

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.subspaces = np.random.randint(
            self.n_features, size=(self.n_members, self.subspace_size)
        )

        self.ensemble = []
        for subspace in self.subspaces:
            clf = clone(self.base_classifier).fit(X[:, subspace], y)
            self.ensemble.append(clf)

        return self

    def predict_proba(self, X):
        esm = np.mean(
            np.array(
                [
                    clf.predict_proba(X[:, self.subspaces[i]])
                    for i, clf in enumerate(self.ensemble)
                ]
            ),
            axis=0,
        )

        return esm

    def predict(self, X):
        pp = self.predict_proba(X)

        y_pred = np.argmax(pp, axis=1)

        return y_pred
