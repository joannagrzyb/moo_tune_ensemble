import os
import numpy as np
import matplotlib.pyplot as plt
from utils.load_dataset import find_datasets


# Plot pareto front scatter function
def scatter_pareto_chart(DATASETS_DIR, n_folds, experiment_name, methods):
    n_rows_p = 1000
    for dataset_id, dataset in enumerate(find_datasets(DATASETS_DIR)):
        print(dataset)
        for clf_id, clf_name in enumerate(methods):
            for fold_id in range(n_folds):
                solutions = []
                for sol_id in range(n_rows_p):
                    try:
                        filename_pareto = "results/%s/pareto_raw/%s/%s/fold%d/sol%d.csv" % (experiment_name, dataset, clf_name, fold_id, sol_id)
                        solution = np.genfromtxt(filename_pareto, dtype=np.float32)
                        solution = solution.tolist()
                        solution[0] = solution[0] * (-1)
                        solution[1] = solution[1] * (-1)
                        solutions.append(solution)
                    except IOError:
                        pass
                if solutions:
                    filename_pareto_chart = "results/%s/pareto_plots/%s/%s/pareto_%s_%s_fold%d" % (experiment_name, dataset, clf_name, dataset, clf_name, fold_id)
                    if not os.path.exists("results/%s/pareto_plots/%s/%s/" % (experiment_name, dataset, clf_name)):
                        os.makedirs("results/%s/pareto_plots/%s/%s/" % (experiment_name, dataset, clf_name))
                    x = []
                    y = []
                    for solution in solutions:
                        x.append(solution[0])
                        y.append(solution[1])
                    x = np.array(x)
                    y = np.array(y)
                    plt.grid(True, color="silver", linestyle=":", axis='both')
                    plt.scatter(x, y, color='black')
                    plt.title("Objective Space", fontsize=12)
                    plt.xlabel('Precision', fontsize=12)
                    plt.ylabel('Recall', fontsize=12)
                    plt.gcf().set_size_inches(6, 3)
                    plt.savefig(filename_pareto_chart+".png", bbox_inches='tight')
                    plt.savefig(filename_pareto_chart+".eps", format='eps', bbox_inches='tight')
                    plt.clf()
                    plt.close()
