import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from rembrandtml.data_providers.data_provider import DataProviderBase

class SkLearnDataProvider(DataProviderBase):
    def __init__(self, dataset_name):
        super(SkLearnDataProvider, self).__init__('sklearn', dataset_name)

    def prepare_data(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        if self.dataset_name == 'mnist-original':
            dataset = self.get_dataset_mnist()
        elif self.dataset_name.lower() == 'boston':
            dataset = self.get_dataset_boston(features, target_feature)
        else:
            raise TypeError(f'The dataset {self.dataset_name} is not supported for {self.name}')
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

    def split(self, X, y, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=test_size, random_state=random_state)
        return ((X_train, y_train), (X_test, y_test))
