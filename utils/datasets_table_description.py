import os
from os import listdir
from os.path import isfile, join
import sys
from scipy.io import arff
import numpy as np
import pandas as pd
import warnings
from load_dataset import load_data, find_datasets

warnings.filterwarnings("ignore")

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#
# DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/test')


# mypath = "../datasets/%s" % directory
# datasets = []
# datasets += ["%s%s" % (directory, os.path.splitext(f)[0]) for f in listdir(mypath) if isfile(join(mypath, f))]


# datasets.sort(key=str.lower)

# print(datasets)





directories = ["9higher_part1/", "9higher_part2/", "9higher_part3/", "9lower/"]

for dir in directories:
    DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s' % dir)
    dir_split = dir.split("/")[0]
    # print("TO", dir_split)
    # print(DATASETS_DIR)
    with open("tables/datasets_%s.tex" % dir_split, "w+") as file:
        for dataset_id, dataset_name in enumerate(find_datasets(DATASETS_DIR)):
            dataset_path = "/home/joannagrzyb/dev/moo_tune_ensemble/datasets/" + dir + dataset_name + ".dat"
            X, y, classes = load_data(dataset_path)
            number_of_features = len(X.columns)
            number_of_objects = len(y)
            # print(X)
            # print(number_of_features, number_of_objects)

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
            # print(majority)

            imbalance_ratio = majority.shape[0]/minority.shape[0]
            # dataset_name = dataset_name.split("/")[1]
            dataset_name = dataset_name.replace("_", "\\_")
            # print(dataset_name)
            print("\\emph{%s} & %0.2g & %d & %d \\\\" % (dataset_name, imbalance_ratio, number_of_objects, number_of_features), file=file)
            # print(ds_name)
            print(dataset_path)


    # for ds_name in datasets:
        # data, meta = arff.loadarff("/home/joannagrzyb/dev/moo_tune_ensemble/datasets/%s.arff" % ds_name)
        # classes = meta[meta.names()[-1]][1]
        # df = pd.DataFrame(data)
        # X = df.iloc[:, 0:-1].values.astype(float)
        # y = df.iloc[:, -1].values.astype(str)
        #
        #
        # number_of_features = len(X[0])
        # number_of_objects = len(y)
        #
        # unique, counts = np.unique(y, return_counts=True)
        # if len(counts) == 1:
        #     raise ValueError("Only one class in procesed data. Use bigger data chunk")
        # elif counts[0] > counts[1]:
        #     majority_name = unique[0]
        #     minority_name = unique[1]
        # else:
        #     majority_name = unique[1]
        #     minority_name = unique[0]
        #
        # minority_ma = np.ma.masked_where(y == minority_name, y)
        # minority = X[minority_ma.mask]
        #
        # majority_ma = np.ma.masked_where(y == majority_name, y)
        # majority = X[majority_ma.mask]
        #
        # imbalance_ratio = majority.shape[0]/minority.shape[0]
        #
        # ds_name = ds_name.split("/")[1]
        # ds_name = ds_name.replace("_", "\\_")
        # print("\\emph{%s} & %0.2g & %d & %d \\\\" % (ds_name, imbalance_ratio, number_of_objects, number_of_features), file=file)
        # print(ds_name)
