import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


DATASETS_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'datasets/test')


def load_data(filename):
    # Load from file
    with open(filename) as file:
        attr_count = 0
        for line in file:
            if line.startswith("@attribute Class"):
                continue
            elif line.startswith("@attribute"):
                attr_count += 1
            elif line.startswith("@data"):
                break

        df = pd.read_csv(file, header=None)

        features = df.iloc[:, 0:-1]
        # Index of the categorical variables in features
        categorical_id = [key for key in dict(features.dtypes) if dict(features.dtypes)[key] in ['object']]
        # Encoding categorical variables into numbers
        encoder = OrdinalEncoder()
        features[categorical_id] = encoder.fit_transform(features[categorical_id])

        labels = df.iloc[:, -1].values.astype(str)
        # Enocoding class names into binary
        class_encoder = LabelEncoder()
        labels = class_encoder.fit_transform(labels)
        classes = np.unique(labels)

        return features, labels, classes


def find_datasets(storage=DATASETS_DIR):
    for f_name in os.listdir(storage):
        yield f_name.split('.')[0]
