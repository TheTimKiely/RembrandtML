import os
import numpy as np
import pandas as pd

from rembrandtml.configuration import RunMode
from rembrandtml.data_providers.data_provider import DataProviderBase

class PandasDataProvider(DataProviderBase):
    def __init__(self, data_config, instrumentation):
        super(PandasDataProvider, self).__init__('pandas', data_config, instrumentation)

    def prepare_data(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        #if self.data_config.dataset_name.lower() == 'titanic':
        dataset = self.get_dataset_from_file(features, self.data_config.dataset_file_path, target_feature)
        #else:
        #    dataset = self.load_from_file(self.data_config.dataset_file_path, features, target_feature)
        # commented out while I try to generalize dataset loading
        #else:
        #    raise TypeError(f'The dataset {self.data_config.dataset_name} is not supported for {self.name}')
        return dataset

    def get_dataset(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        return dataset

    def get_features(self, df, features):
        X = []
        for feature in features:
            X.append(df[feature])
        return tuple(X)

    def preprocess_titanic_data(self, df, features):
        import  re

        if features is None:
            raise

        title_na = 0
        fare_na = 0

        data = pd.DataFrame()

        if 'Has_Cabin' in features:
            data['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

        if 'Deck' in features:
            deck = {"U": 1, "C": 2, "B": 3, "D": 4, "E": 5, "F": 6, "A": 7, "G": 8}
            df['Cabin']= df['Cabin'].fillna("U0")
            df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
            df['Deck'] = df['Deck'].map(deck)
            df['Deck'] = df['Deck'].fillna(0)
            data['Deck'] = df['Deck'].astype(int)

        if 'Family_Size' in features:
            data['Family_Size'] = df['SibSp'] + df['Parch'] + 1

        if 'Alone' in features:
            df['relatives'] = df['SibSp'] + df['Parch']
            df.loc[df['relatives'] > 0, 'alone'] = 0
            df.loc[df['relatives'] == 0, 'alone'] = 1
            data['Alone'] = df['alone'].astype(int)

        if 'Port' in features:
            embarked_na = 'S'
            ports = {'S': 0, 'C': 1, 'Q': 2}
            df['Embarked'] = df['Embarked'].fillna(embarked_na)
            data['Port'] = df['Embarked'].map(ports)

        # Convert category features
        if 'Sex' in features:
            genders = {'male': 0, 'female': 1}
            data['Sex'] = df['Sex'].map(genders)

        if 'Title' in features:
            titles = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
            df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
            df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                                'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
            df['Title'] = df['Title'].replace('Mlle', 'Miss')
            df['Title'] = df['Title'].replace('Ms', 'Miss')
            df['Title'] = df['Title'].replace('Mme', 'Mrs')
            # convert titles into numbers
            df['Title'] = df['Title'].map(titles)
            # filling NaN with 0, to get safe
            data['Title'] = df['Title'].fillna(title_na)

        if 'Age' in features:
            mean = df["Age"].mean()
            std = df["Age"].std()
            is_null = df["Age"].isnull().sum()
            # compute random numbers between the mean, std and is_null
            rand_age = np.random.randint(mean - std, mean + std, size=is_null)
            # fill NaN values in Age column with random values generated
            age_slice = df["Age"].copy()
            age_slice[np.isnan(age_slice)] = rand_age
            df["Age"] = age_slice
            df['Age'] = df['Age'].astype(int)
            df.loc[df['Age'] <= 11, 'Age'] = 0
            df.loc[(df['Age'] > 11) & (df['Age'] <= 22), 'Age'] = 1
            df.loc[(df['Age'] > 22) & (df['Age'] <= 33), 'Age'] = 2
            df.loc[(df['Age'] > 33) & (df['Age'] <= 44), 'Age'] = 3
            df.loc[(df['Age'] > 44) & (df['Age'] <= 55), 'Age'] = 4
            df.loc[(df['Age'] > 55) & (df['Age'] <= 66), 'Age'] = 5
            df.loc[df['Age'] > 66, 'Age'] = 6
            data['Age'] = df['Age']

        ''' Pandas Interval doesn't enumerate propertly
        df['Fare'] = df.Fare.astype(int)
        fare_bands = pd.qcut(df['Fare'], 6)
        for fare_band in fare_bands:
            print(f'left: {fare_band.left} right: {fare_band.right}')
        for i, fare_band in enumerate(fare_bands):
            print(f'i: {i} left: {fare_band.left} right: {fare_band.right}')
            df.loc[(df['Fare'] >= fare_band.left) & (df['Fare'] < fare_band.right), 'Fare'] = i
        '''

        if 'Fare' in features:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            df['Fare'] = df['Fare'].astype(int)
            df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
            df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
            df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
            df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare'] = 3
            df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare'] = 4
            df.loc[df['Fare'] > 250, 'Fare'] = 5
            data['Fare'] = df['Fare'].astype(int)

        if 'Pclass' in features:
            data['Pclass'] = df['Pclass']

        if 'SibSp' in features:
            data['SibSp'] = df['SibSp']

        if 'Parch' in features:
            data['Parch'] = df['Parch']

        # Computed features
        if 'Age_Class' in features:
            mean = df["Age"].mean()
            std = df["Age"].std()
            is_null = df["Age"].isnull().sum()
            # compute random numbers between the mean, std and is_null
            rand_age = np.random.randint(mean - std, mean + std, size=is_null)
            # fill NaN values in Age column with random values generated
            age_slice = df["Age"].copy()
            age_slice[np.isnan(age_slice)] = rand_age
            df["Age"] = age_slice
            df['Age'] = df['Age'].astype(int)
            data['Age_Class'] = df['Age'] * df['Pclass']

        if 'Fare_Per_Person' in features:
            df.Fare = df['Fare'].fillna(fare_na)
            df['Fare_Per_Person'] = df['Fare'] / (df['relatives'] + 1)
            data['Fare_Per_Person'] = df['Fare_Per_Person'].astype(int)

        return (data.columns.values, data.values)

    def get_column_values(self, file_path, column_name):
        df = pd.read_csv(file_path)
        return df[column_name]

    def get_prediction_data(self, features, prediction_file):
        # y_pred will be None because we are getting the prediction data
        X_columns, X_pred, y_pred = self.get_dataset_from_file(features, prediction_file, mode=RunMode.PREDICT)
        return X_pred

    #def get_dataset_kaggle(self, features, target_feature):
    #    return self.get_dataset_from_file(features, self.data_config.dataset_file_path, target_feature)

    def get_dataset_from_file(self, features, file_name, target_feature = None, mode=RunMode.TRAIN):
        """
        Loads a file into a pandas DataFrame and then parses out columns based on the 'features' and 'target_feature' parameters
        :param path: The platform-agnostic path to the data file to be loaded
        :param features: An interable of features to be used from the dataset.  If 'None', all features will be used
        :param target_feature: The name of the column that contains label data.
        :return: A tuple of X_columns, X, y
        """

        df = pd.read_csv(file_name, sep=self.data_config.file_separator)

        # Populate y
        if mode is RunMode.TRAIN or mode is RunMode.EVALUATE:
            if target_feature:
                y = df[target_feature]
            else:
                if 'target' in df.keys():
                    y = df['target']
                else:
                    raise TypeError(f'Could not find target column.  Either the \'target_feature\' parameter must be supplied or the dataset must have a column named \'target\'')
        else:
            y = None

        # Populate X
        if 'titanic' in self.data_config.dataset_name.lower():
            (X_columns, X) = self.preprocess_titanic_data(df, features=features)
        elif features:
            X = self.get_features(df, features)
        else:
            X = df.drop(target_feature, 1)

        #y = y.reshape(-1, 1)
        dataset = (X_columns, X, y)
        return dataset