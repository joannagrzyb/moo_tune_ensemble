import numpy as np
import strlearn as sl
import warnings
import os
import time

from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler

from utils.load_dataset import load_data, find_datasets
from methods.moo_ensemble import MooEnsembleSVC
from methods.random_subspace_ensemble import RandomSubspaceEnsemble

warnings.filterwarnings("ignore")
start = time.time()

base_estimator = SVC(probability=True)
methods = {
    "MooEnsembleSVC": MooEnsembleSVC(base_classifier=base_estimator),
    "RandomSubspace": RandomSubspaceEnsemble(base_classifier=base_estimator),
}

# Repeated Stratified K-Fold cross validator
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
n_folds = n_splits * n_repeats


"""
Imbalanced datasets from KEEL:
    - IR (imbalance ratio) = majority / minority
        higher IR, higher imbalance of the dataset
"""
DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/test')

metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.geometric_mean_score_1, sl.metrics.geometric_mean_score_2, sl.metrics.f1_score, sl.metrics.recall, sl.metrics.specificity, sl.metrics.precision]
metrics_alias = ["BAC", "Gmean", "Gmean2", "F1score", "Recall", "Specificity", "Precision"]

for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
    start_ds = time.time()
    print("Dataset: ", dataset_id, dataset)
    dataset_path = "datasets/9lower/" + dataset + ".dat"
    X, y, classes = load_data(dataset_path)
    X = X.to_numpy()
    # Normalization - transform data to [0, 1]
    X = MinMaxScaler().fit_transform(X, y)

    scores = np.zeros((len(metrics), len(methods), n_folds))
    diversity = np.zeros((len(methods), n_folds, 4))

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
            calculate_diversity = getattr(clf, "calculate_diversity", None)
            if callable(calculate_diversity):
                diversity[clf_id, fold_id] = clf.calculate_diversity()
            else:
                diversity[clf_id, fold_id] = None
            print(diversity[clf_id, fold_id])
    print(scores)
    # Save results to csv
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            # Save metric results
            filename = "results/experiment_test/9lower/raw_results/%s/%s/%s.csv" % (metric, dataset, clf_name)
            if not os.path.exists("results/experiment_test/9lower/raw_results/%s/%s/" % (metric, dataset)):
                os.makedirs("results/experiment_test/9lower/raw_results/%s/%s/" % (metric, dataset))
            np.savetxt(fname=filename, fmt="%f", X=scores[metric_id, clf_id, :])
        # Save diversity results
        filename = "results/experiment_test/9lower/diversity_results/%s/%s.csv" % (dataset, clf_name)
        if not os.path.exists("results/experiment_test/9lower/diversity_results/%s/" % (dataset)):
            os.makedirs("results/experiment_test/9lower/diversity_results/%s/" % (dataset))
        np.savetxt(fname=filename, fmt="%f", X=diversity[clf_id, :, :])

    end_ds = time.time()
    print("TIME: %.0f sec" % (end_ds - start_ds))

end = time.time()
print("TIME: %.0f sec" % (end - start))
