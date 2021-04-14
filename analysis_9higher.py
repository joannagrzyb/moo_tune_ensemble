import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import chi2

from methods.moo_ensemble import MooEnsembleSVC
from methods.moo_ensemble_bootstrap import MooEnsembleSVCbootstrap
from methods.random_subspace_ensemble import RandomSubspaceEnsemble
from methods.feature_selection_clf import FeatueSelectionClf
from utils.load_dataset import find_datasets
from utils.plots import scatter_pareto_chart
from utils.wilcoxon_ranking import pairs_metrics_multi


# DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/9higher_part1')

base_estimator = {'SVM': SVC(probability=True)}

methods = {
    "MooEnsembleSVC": MooEnsembleSVC(base_classifier=base_estimator),
    "MooEnsembleSVCbootstrap": MooEnsembleSVCbootstrap(base_classifier=base_estimator),
    "RandomSubspace": RandomSubspaceEnsemble(base_classifier=base_estimator),
    "SVM": SVC(),
    "FS": FeatueSelectionClf(base_estimator, chi2),
    "FSIRSVM": 0
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

# Load data from file
n_datasets = 0
datasets = []
directories = ["9higher_part1", "9higher_part2", "9higher_part3"]
for dir_id, dir in enumerate(directories):
    DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s/' % dir)
    n_datasets += len(list(enumerate(find_datasets(DATASETS_DIR))))
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        datasets.append(dataset)

data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))

for dir_id, dir in enumerate(directories):
    dir_id += 1
    for dataset_id, dataset in enumerate(datasets):
        for clf_id, clf_name in enumerate(methods):
            for metric_id, metric in enumerate(metrics_alias):
                try:
                    filename = "results/experiment_server/experiment%d_%s/raw_results/%s/%s/%s.csv" % (dir_id, dir, metric, dataset, clf_name)
                    if not os.path.isfile(filename):
                        # print("File not exist - %s" % filename)
                        continue
                    scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    data_np[dataset_id, metric_id, clf_id] = scores
                except:
                    print("Error loading dataset - %s!" % dataset)


# Wilcoxon ranking - statistic test for methods: SEMOOS and SEMOOSb
pairs_metrics_multi(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment_9higher", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot", ref_method=methods_alias[0])

pairs_metrics_multi(method_names=methods_alias, data_np=data_np, experiment_name="experiment_server/experiment_9higher", dataset_names=datasets, metrics=metrics_alias, filename="ex9h_ranking_plot", ref_method=methods_alias[1])
