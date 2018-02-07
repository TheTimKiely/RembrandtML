import numpy as np
from sklearn.model_selection import train_test_split

from rembrandtml.entities import MLEntityBase


class DataProviderBase(MLEntityBase):
    def __init__(self, name, data_config, instrumentation):
        super(DataProviderBase, self).__init__(instrumentation)
        self.name = name
        self.data_config = data_config

    def split(self, X, y, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=test_size, random_state=random_state)
        return ((X_train, y_train), (X_test, y_test))

    def get_dataset(self):
        pass

    def prepare_data(self, features=None, target_feature=None, sample_size=None):
        # Retrieve X and y from dataset
        return

        # 2. get dataset
        dataset = self.get_dataset()
        if self.name.lower() == 'pandas':
            self.prepare_data_pandas(features, target_feature)
        elif self.name.lower() == 'sklearn':
            self.prepare_data_sklearn(features, sample_size)
        else:
            raise TypeError(f'Unsupported data provider {self.name}')

    def prepare_data_pandas(self, features, target_feature):
        import pandas as pd
        if (self.data_config.dataset_name.lower() == 'gapminder'):
            # ToDo the file name is a 'magic value' that belongs in a config file
            data_dir = os.path.join(self.Base_Directory, 'data', 'gapminder')
            gapminder = pd.read_csv(os.path.join(data_dir, 'gm_2008_region.csv'))
            self.prepare_dataset(gapminder, features)
        else:
            raise TypeError(
                f'The dataset: {self.data_config.dataset_name} is not supported in the framework: {self.framework_name}')

    def prepare_data_sklearn(self, features, sample_size):
        if (self.data_config.dataset_name.lower() == 'imdb'):
            self.prepare_imdb_data()
        elif (self.data_config.dataset_name.lower() == 'jena_climate'):
            self.prepare_climage_data(sample_size)
        elif self.data_config.dataset_name.lower() == 'boston' or \
                self.data_config.dataset_name.lower() == 'mnist-original':
            from sklearn import datasets
            # boston = pd.read_csv('boston.csv')
            # X = boston.drop('MEDV', axis=1).values
            # y = boston['MEDV'].values

            boston = datasets.load_boston()
            self.prepare_dataset(boston, features)
            data = boston['data']
            if features:
                indeces = [i for i, k in enumerate(boston['feature_names']) if k in features]
                X_rooms = data[:, 5]
                X = np.zeros((data.shape[0], len(indeces)))
                for i, index in enumerate(indeces):
                    X[:, i] = data[:, index]
            else:
                X = data
            y = boston['target']
            self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            # ToDo: why reshape
            # 1. What is the shape supposed to be?
            # 2. How do we know the required shape?

            self.y_train = y_train.reshape(-1, 1)
            self.y_test = y_test.reshape(-1, 1)
            # self.X = X_rooms.reshape(-1, 1)
        elif self.data_config.dataset_name.lower() == 'ca_housing':
            housing = datasets.fetch_california_housing()
            self.prepare_dataset(housing, features)
        else:
            raise TypeError(self.log(f'The dataset {self.data_config.dataset_name} is not supported'))
        return ((self.X_train, self.y_train), (self.X_test, self.y_test))

    def prepare_dataset(self, dataset, features):
        '''
        Generic method to initialize self.X_train, self.X_test, self.y_train, and self.y_test
        :param dataset: The dataset that will be used.
        :param features: A tuple of the names of the features to retrive.  If None, all features will be retrieved.
        :return: None
        '''
        self.log(
            f'Retrieving data from dataset: {dataset} for {features + "features: " if features else "all features"}')
        data = dataset['data']
        if features:
            indeces = [i for i, k in enumerate(dataset['feature_names']) if k in features]
            X = np.zeros((data.shape[0], len(indeces)))
            for i, index in enumerate(indeces):
                X[:, i] = data[:, index]
        else:
            X = data
        y = dataset['target']
        self.X_train, self.X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # ToDo: why reshape
        # 1. What is the shape supposed to be?
        # 2. How do we know the required shape?

        self.y_train = y_train.reshape(-1, 1)
        self.y_test = y_test.reshape(-1, 1)

    def check_scale(self):
        # Import scale
        from sklearn.preprocessing import scale

        # Scale the features: X_scaled
        self.X_scaled = scale(self.X)

        # Print the mean and standard deviation of the unscaled features
        print("Mean of Unscaled Features: {}".format(np.mean(self.X)))
        print("Standard Deviation of Unscaled Features: {}".format(np.std(self.X)))

        # Print the mean and standard deviation of the scaled features
        print("Mean of Scaled Features: {}".format(np.mean(self.X_scaled)))
        print("Standard Deviation of Scaled Features: {}".format(np.std(self.X_scaled)))

    def scale(self, algorith='standardization'):
        pass
