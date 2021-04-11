import os
import numpy as np
import warnings
from load_dataset import load_data, find_datasets


warnings.filterwarnings("ignore")

directories = ["9higher_part1/", "9higher_part2/", "9higher_part3/", "9lower/"]

for dir in directories:
    DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s' % dir)
    dir_split = dir.split("/")[0]

    with open("tables/datasets_%s.tex" % dir_split, "w+") as file:
        for dataset_id, dataset_name in enumerate(find_datasets(DATASETS_DIR)):
            dataset_path = "/home/joannagrzyb/dev/moo_tune_ensemble/datasets/" + dir + dataset_name + ".dat"
            X, y, classes = load_data(dataset_path)
            number_of_features = len(X.columns)
            number_of_objects = len(y)

            unique, counts = np.unique(y, return_counts=True)

            if len(counts) == 1:
                raise ValueError("Only one class in procesed data.")
            elif counts[0] > counts[1]:
                majority_name = unique[0]
                minority_name = unique[1]
            else:
                majority_name = unique[1]
                minority_name = unique[0]

            minority_ma = np.ma.masked_where(y == minority_name, y)
            minority = X[minority_ma.mask]

            majority_ma = np.ma.masked_where(y == majority_name, y)
            majority = X[majority_ma.mask]

            imbalance_ratio = majority.shape[0]/minority.shape[0]
            dataset_name = dataset_name.replace("_", "\\_")
            print("\\emph{%s} & %0.2g & %d & %d \\\\" % (dataset_name, imbalance_ratio, number_of_objects, number_of_features), file=file)
            print(dataset_path)
