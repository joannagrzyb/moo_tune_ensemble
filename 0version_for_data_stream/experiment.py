import numpy as np
import strlearn as sl
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.ndimage import gaussian_filter1d
import warnings

from methods.moo_ensemble import MooEnsembleSVC
from methods.moo_ensemble_all import MooEnsembleAllSVC
from methods.random_subspace_ensemble import RandomSubspaceEnsemble

warnings.filterwarnings("ignore")

base_estimator = SVC(probability=True)

clfs = {
    "MooEnsembleSVC": MooEnsembleSVC(base_classifier=base_estimator),
    # !!! nie działa
    # "MooEnsembleAllSVC": MooEnsembleAllSVC(base_classifier=base_estimator),
    "RandomSubspace": RandomSubspaceEnsemble(base_classifier=base_estimator),
    # "MLPC": MLPClassifier(hidden_layer_sizes=(10)),
}

stream = sl.streams.StreamGenerator(n_chunks=50, chunk_size=500, n_features=7, n_informative=3, n_redundant=4, y_flip=0.05, n_drifts=1, weights=[0.8, 0.2], random_state=1111)

metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.geometric_mean_score_1, sl.metrics.geometric_mean_score_2, sl.metrics.f1_score, sl.metrics.recall, sl.metrics.specificity, sl.metrics.precision]

evaluator = sl.evaluators.TestThenTrain(metrics, verbose=True)
evaluator.process(stream, clfs.values())

labels = list(clfs.keys())

linestyles = ['-'] * len(clfs)
# linestyles = ['-', '-', '--']
for m, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    plt.suptitle(metric.__name__)
    # plt.ylim(0, 1)
    for i, clf in enumerate(clfs):
        plt.plot(gaussian_filter1d(evaluator.scores[i, :, m], 2), label=labels[i], linestyle=linestyles[i])
    plt.legend()
    plt.savefig("results/sl_%s.png" % metric.__name__)
    # plt.show()
