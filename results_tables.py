import os
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from methods.moo_ensemble import MooEnsembleSVC
from methods.moo_ensemble_bootstrap import MooEnsembleSVCbootstrap
from methods.random_subspace_ensemble import RandomSubspaceEnsemble
from methods.feature_selection_clf import FeatueSelectionClf
from utils.load_dataset import find_datasets


warnings.filterwarnings("ignore")


base_estimator = {'SVM': SVC(probability=True)}
# IR is an example, not real values of datasets
IR = {0: 1, 1: 1}

methods = {
    "MooEnsembleSVC": MooEnsembleSVC(base_classifier=base_estimator),
    "MooEnsembleSVCbootstrap": MooEnsembleSVCbootstrap(base_classifier=base_estimator),
    "RandomSubspace": RandomSubspaceEnsemble(base_classifier=base_estimator),
    "SVM": SVC(),
    "FS": FeatueSelectionClf(base_estimator, chi2),
    "FSIRSVM": FeatueSelectionClf(SVC(kernel='linear', class_weight=IR), chi2)
}

methods_alias = [
                "SEMOOS",
                "SEMOOSb",
                "RS",
                "SVM",
                "FS",
                "FSIRSVM"
                ]

metrics_alias = ["BAC", "Gmean", "Gmean2", "F1score", "Recall", "Specificity", "Precision"]

n_splits = 2
n_repeats = 5
n_folds = n_splits * n_repeats
n_methods = len(methods_alias) * len(base_estimator)
n_metrics = len(metrics_alias)

directories = ["9higher_part1", "9higher_part2", "9higher_part3", "9lower"]

for metric_id, metric in enumerate(metrics_alias):
    with open("results/tables/results_%s.tex" % metric, "w+") as file:
        print()
        for dir_id, dir in enumerate(directories):
            dir_id += 1
            DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s/' % dir)
            n_datasets = len(list(enumerate(find_datasets(DATASETS_DIR))))
            data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
            mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
            stds = np.zeros((n_datasets, n_metrics, n_methods))
            for dataset_id, dataset_name in enumerate(find_datasets(DATASETS_DIR)):
                for clf_id, clf_name in enumerate(methods):
                    try:
                        filename = "results/experiment_server/experiment%d_%s/raw_results/%s/%s/%s.csv" % (dir_id, dir, metric, dataset_name, clf_name)
                        scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                        mean_score = np.mean(scores)
                        mean_scores[dataset_id, metric_id, clf_id] = mean_score
                        std = np.std(scores)
                        stds[dataset_id, metric_id, clf_id] = std
                    except:
                        print("Error loading data!")
                        print(filename)

                dataset_name = dataset_name.replace("_", "\\_")
                print("\\emph{%s} & %0.2f $\\pm$ %0.2f & %0.2f $\\pm$ %0.2f & %0.2f $\\pm$ %0.2f & %0.2f $\\pm$ %0.2f & %0.2f $\\pm$ %0.2f & %0.2f $\\pm$ %0.2f \\\\" % (
                    dataset_name,
                    mean_scores[dataset_id, metric_id, 0], stds[dataset_id, metric_id, 0],
                    mean_scores[dataset_id, metric_id, 1], stds[dataset_id, metric_id, 0],
                    mean_scores[dataset_id, metric_id, 2], stds[dataset_id, metric_id, 0],
                    mean_scores[dataset_id, metric_id, 3], stds[dataset_id, metric_id, 0],
                    mean_scores[dataset_id, metric_id, 4], stds[dataset_id, metric_id, 0],
                    mean_scores[dataset_id, metric_id, 5], stds[dataset_id, metric_id, 0]
                    ), file=file)
            print("\\hline", file=file)
