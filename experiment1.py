import numpy as np
import strlearn as sl

from sklearn.svm import SVC
import warnings
import os

from methods.moo_ensemble import MooEnsembleSVC
# from methods.moo_ensemble_all import MooEnsembleAllSVC
from methods.random_subspace_ensemble import RandomSubspaceEnsemble
from load_dataset import load_data, find_datasets
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
import time

start = time.time()

warnings.filterwarnings("ignore")




# Jaką wybrać liczbę cech do testowania w mojej metodzie?

base_estimator = SVC(probability=True)

methods = {
    "MooEnsembleSVC": MooEnsembleSVC(base_classifier=base_estimator),
    # !!! nie działa
    # "MooEnsembleAllSVC": MooEnsembleAllSVC(base_classifier=base_estimator),
    "RandomSubspace": RandomSubspaceEnsemble(base_classifier=base_estimator),
}

# Repeated Stratified K-Fold cross validator
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
n_folds = n_splits * n_repeats

# DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/9higher_part1')
DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/1_5_9')

# DATASETS_DIR = [
#         os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/9higher_part1'),
#         os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/1_5_9')
# ]

"""
Imbalanced datasets from KEEL:
    - Imbalance ratio higher than 9 - Part I
    - IR (imbalance ratio) = majority / minority
        higher IR, higher imbalance of the dataset
"""

metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.geometric_mean_score_1, sl.metrics.geometric_mean_score_2, sl.metrics.f1_score, sl.metrics.recall, sl.metrics.specificity, sl.metrics.precision]
metrics_alias = ["BAC", "Gmean", "Gmean2", "F1score", "Recall", "Specificity", "Precision"]

for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    print("Dataset: ", dataset_id, dataset)
    # dataset_path = "datasets/9higher_part1/" + dataset + ".dat"
    dataset_path = "datasets/1_5_9/" + dataset + ".dat"
    X, y, classes = load_data(dataset_path)
    X = X.to_numpy()

    # Normalization - transform data to [0, 1]
    # X = MinMaxScaler().fit_transform(X, y)

    scores = np.zeros((len(metrics), len(methods), n_folds))

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        for clf_id, clf_name in enumerate(methods):
            print(f"Fold number: {fold_id}, clf: {clf_name}")
            clf = clone(methods[clf_name])
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            for metric_id, metric in enumerate(metrics):
                scores[metric_id, clf_id, fold_id] = metric(y_test, y_pred)
                print(metric(y_test, y_pred))
    print(scores)
    # # Save results to csv - 9higher_part1
    # for clf_id, clf_name in enumerate(methods):
    #     for metric_id, metric in enumerate(metrics_alias):
    #         filename = "results/experiment1/9higher_part1/raw_results/%s/%s/%s.csv" % (metric, dataset, clf_name)
    #         if not os.path.exists("results/experiment1/9higher_part1/raw_results/%s/%s/" % (metric, dataset)):
    #             os.makedirs("results/experiment1/9higher_part1/raw_results/%s/%s/" % (metric, dataset))
    #         np.savetxt(fname=filename, fmt="%f", X=scores[metric_id, clf_id, :])

    # Save results to csv
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            filename = "results/experiment1/1_5_9/raw_results/%s/%s/%s.csv" % (metric, dataset, clf_name)
            if not os.path.exists("results/experiment1/1_5_9/raw_results/%s/%s/" % (metric, dataset)):
                os.makedirs("results/experiment1/1_5_9/raw_results/%s/%s/" % (metric, dataset))
            np.savetxt(fname=filename, fmt="%f", X=scores[metric_id, clf_id, :])

    end = time.time()
    print("TIME: %.0f sec" % (end - start))

end = time.time()
print("TIME: %.0f sec" % (end - start))
