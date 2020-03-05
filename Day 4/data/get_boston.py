"""
Boston Housing Dataset is one of the built-in datasets for sklearn, but may
reqiure internet access.

This allows for reproducable construction of the boston housing dataset for offline use.
"""
import pickle

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd

def pickle_boston_dataframe():
    boston = load_boston()

    data = boston.data
    target = boston.target.reshape(-1,1)
    names = boston.feature_names

    data_all = np.concatenate([data, target], axis=1)
    name_all = np.concatenate([names, ['MEDV']], axis=0)

    offline_storage = {
        'dataframe': pd.DataFrame(data=data_all, columns=name_all),
        'description': boston.DESCR
    }

    pickle.dump(offline_storage, open('boston_housing.pickle', 'wb'))

pickle_boston_dataframe()
