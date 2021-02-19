import numpy as np
import strlearn as sl
import autograd.numpy as anp
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from pymoo.model.problem import Problem


# !!! NIE DZIAŁA
class OptimizationParamAll(Problem):
    def __init__(self, X, y, test_size, estimator, scale_features, n_features, n_classifiers=10, n_param=2, objectives=2, random_state=0, feature_names=None):

        self.estimator = estimator
        self.test_size = test_size
        self.objectives = objectives
        self.n_classifiers = n_classifiers
        self.n_param = n_param
        self.scale_features = scale_features
        self.n_features = n_features
        self.ensemble = []
        self.X = X

        # If test size is not specify or it is 0, everything is took to test and train
        if self.test_size != 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = np.copy(X), np.copy(y), np.copy(X), np.copy(y)

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        xl_real = [1E6, 1E-7]
        xl_binary = [0] * self.n_features
        xl_one = np.hstack([xl_real, xl_binary])
        xu_real = [1E9, 1E-4]
        xu_binary = [1] * n_features
        xu_one = np.hstack([xu_real, xu_binary])
        xl = []
        xu = []
        for n in range(self.n_classifiers):
            xl = np.hstack([xl, xl_one])
            xu = np.hstack([xu, xu_one])

        n_variable = (self.n_param + self.n_features) * self.n_classifiers
        super().__init__(n_var=n_variable, n_obj=objectives,
                         n_constr=1, elementwise_evaluation=True, xl=xl, xu=xu)

    # def predict(self, X):
    #     # Prediction based on the average support vectors
    #     ens_sup_matrix = self.ensemble_support_matrix(X)
    #     average_support = np.mean(ens_sup_matrix, axis=0)
    #     prediction = np.argmax(average_support, axis=1)
    #     # Return prediction
    #     return self.classes_[prediction]

    # x: a two dimensional matrix where each row is a point to evaluate and each column a variable
    def validation(self, x):
        print("x", x)
        # C_list: list of hyperparameters C
        C_list = x[0::(self.n_features+self.n_param)]
        # gamma_list: list of hyperparameters gamma
        gamma_list = x[1::(self.n_features+self.n_param)]
        # list of selected features for each model
        next = (self.n_features+self.n_param)
        selected_features = []
        start = 2
        stop = next
        step = 1
        for n in range(self.n_classifiers):
            sf = x[start:stop:step]
            selected_features.append(sf.tolist())
            start += next
            stop += next

        # If at least one element in x are True
        for i, sf in enumerate(selected_features):
            if True in sf:
                clf = clone(self.estimator.set_params(C=C_list[i], gamma=gamma_list[i]))
                clf.fit(self.X_train[:, sf], self.y_train)
                self.ensemble.append(clf)
# ???
        print(self.ensemble)
        for i, sf in enumerate(selected_features):
            if True in sf:
                for clf_id, clf in enumerate(self.ensemble):
                    y_pred = clf.predict(self.X_test[:, sf[clf_id]])
                    metrics = [sl.metrics.precision(self.y_test, y_pred), sl.metrics.recall(self.y_test, y_pred)]
            else:
                metrics = [0, 0]

        return metrics

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)
        # print("x", x)
        # Function F is always minimize, but the minus sign (-) before F means maximize
        f1 = -1 * scores[0]
        f2 = -1 * scores[1]
        out["F"] = anp.column_stack(np.array([f1, f2]))

        # Function constraint to select specific numbers of features:
        number = int((1 - self.scale_features) * self.n_features)
        out["G"] = (self.n_features - np.sum(x[2:]) - number) ** 2

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
