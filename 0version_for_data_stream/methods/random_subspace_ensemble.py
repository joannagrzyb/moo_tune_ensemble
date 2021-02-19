import numpy as np
import strlearn as sl
from random import sample
from sklearn.base import clone
from sklearn.base import BaseEstimator


class RandomSubspaceEnsemble(BaseEstimator):

    def __init__(self, base_classifier, n_classifiers=10):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.selected_features = []
        self.ensemble = []

    def partial_fit(self, X, y, classes=None):
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Select random subset of features
        n_features = X.shape[1]
        n_selected_features = int(n_features/2)
        sf = sample(range(0, X.shape[1]), n_selected_features)
        self.selected_features.append(sf)
        # Train new candidate on selected features and add to ensemble
        candidate = clone(self.base_classifier).fit(X[:, sf], y)
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
