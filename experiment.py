import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from scipy.ndimage import gaussian_filter1d

from methods.moo_ensemble import MooEnsembleSVC
import warnings

warnings.filterwarnings("ignore")

base_estimator = SVC(probability=True)

clfs = {
    "MooEnsembleSVC": MooEnsembleSVC(base_estimator),
    # "L++CDS": sl.ensembles.LearnppCDS(base_estimator),
    # "KMC": sl.ensembles.KMC(base_estimator),
    # "L++NIE": sl.ensembles.LearnppNIE(base_estimator),
    # "REA": sl.ensembles.REA(base_estimator),
    # "OUSE": sl.ensembles.OUSE(base_estimator),
    # "MLPC": MLPClassifier(hidden_layer_sizes=(10))
}

stream = sl.streams.StreamGenerator(n_chunks=10, chunk_size=500, n_features=5, n_informative=2, n_redundant=3, y_flip=0.05, n_drifts=0, weights=[0.8, 0.2], random_state=1111)



metrics = [sl.metrics.balanced_accuracy_score, sl.metrics.geometric_mean_score_1, sl.metrics.geometric_mean_score_2, sl.metrics.f1_score, sl.metrics.recall, sl.metrics.specificity, sl.metrics.precision]
# metrics = [sl.metrics.binary_confusion_matrix]

evaluator = sl.evaluators.TestThenTrain(metrics, verbose=True)
evaluator.process(stream, clfs.values())


labels = list(clfs.keys())
linestyles = ['-']
# ,'-','-','-','-','-','-','-','-','-','--','--','--','--','--','--','--','--','--','--']
for m, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    plt.suptitle(metric.__name__)
    # plt.ylim(0, 1)
    for i, clf in enumerate(clfs):
        plt.plot(gaussian_filter1d(evaluator.scores[i, :, m], 2), label=labels[i], linestyle=linestyles[i])
        # plt.plot(evaluator.scores[i, :, m], label=labels[i])
    plt.legend()
    plt.savefig("results/sl_%s.png" % metric.__name__)
    # plt.show()
