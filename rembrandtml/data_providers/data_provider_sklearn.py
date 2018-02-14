import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from rembrandtml.data_providers.data_provider import DataProviderBase

class SkLearnDataProvider(DataProviderBase):
    def __init__(self, data_config, instrumentation):
        super(SkLearnDataProvider, self).__init__('sklearn', data_config, instrumentation)
        self.dataset_map = {'boston': datasets.load_boston(), 'iris': datasets.load_iris(), 'diabetes': datasets.load_diabetes()}

    def prepare_data(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        if self.data_config.dataset_name.lower() in self.dataset_map.keys():
            dataset = self.get_dataset(self.data_config.dataset_name, features, target_feature)
        elif self.data_config.dataset_name == 'mnist-original':
            dataset = self.get_dataset_mnist()
        elif self.data_config.dataset_name.lower() == 'boston':
            dataset = self.get_dataset_boston(features, target_feature)
        elif self.data_config.dataset_name[0:6].lower() == 'kaggle':
            dataset = self.get_dataset_kaggle(features, target_feature)
        else:
            error = f'The dataset {self.data_config.dataset_name} is not supported for {self.name}.'
            raise TypeError(error)
        return dataset

    def get_dataset(self, name, features, target_feature):
        data = self.dataset_map[name]
        X_columns = data.feature_names
        X = data.data
        y = data.target
        dataset = (X_columns, X, y)
        return dataset

    def get_dataset_mnist(self):
        mnist = datasets.fetch_mldata('MNIST original')
        y = mnist['target'].reshape(-1, 1)
        return (mnist['data'], y)

    def get_dataset_boston(self, features, target_feature):
        boston = datasets.load_boston()
        y = boston['target'].reshape(-1, 1)
        data = boston['data']
        if features:
            indeces = [i for i, k in enumerate(boston['feature_names']) if k in features]
            X = np.zeros((data.shape[0], len(indeces)))
            for i, index in enumerate(indeces):
                X[:, i] = data[:, index]
        else:
            X = data

        return (X, y)

