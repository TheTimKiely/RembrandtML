import os
import numpy as np
import pandas as pd
from rembrandtml.data_providers.data_provider import DataProviderBase

class PandasDataProvider(DataProviderBase):
    def __init__(self, data_config, instrumentation):
        super(PandasDataProvider, self).__init__('pandas', data_config, instrumentation)

    def prepare_data(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        if self.data_config.dataset_name[0:6].lower() == 'kaggle':
            dataset = self.get_dataset_kaggle(features, target_feature)
        else:
            raise TypeError(f'The dataset {self.data_config.dataset_name} is not supported for {self.name}')
        return dataset

    def get_dataset(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        return dataset

    def get_features(self, df, features):
        X = []
        for feature in features:
            X.append(df[feature])
        return tuple(X)

    def preprocess_titanic_data(self, df):
        import  re

        title_na = 0
        fare_na = 0

        deck = {"U": 1, "C": 2, "B": 3, "D": 4, "E": 5, "F": 6, "A": 7, "G": 8}
        df['Cabin']= df['Cabin'].fillna("U0")
        df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
        df['Deck'] = df['Deck'].map(deck)
        df['Deck'] = df['Deck'].fillna(0)
        df['Deck'] = df['Deck'].astype(int)
        df = df.drop('Cabin', axis=1)

        mean = df["Age"].mean()
        std = df["Age"].std()
        is_null = df["Age"].isnull().sum()
        # compute random numbers between the mean, std and is_null
        rand_age = np.random.randint(mean - std, mean + std, size=is_null)
        # fill NaN values in Age column with random values generated
        age_slice = df["Age"].copy()
        age_slice[np.isnan(age_slice)] = rand_age
        df["Age"] = age_slice
        df["Age"] = df["Age"].astype(int)

        if self.data_config.parameters['create_alone_column']:
            df['relatives'] = df['SibSp'] + df['Parch']
            df.loc[df['relatives'] > 0, 'alone'] = 0
            df.loc[df['relatives'] == 0, 'alone'] = 1
            df['alone'] = df['alone'].astype(int)

        embarked_na = 'S'
        df['Embarked'] = df['Embarked'].fillna(embarked_na)

        df.Fare = df['Fare'].fillna(fare_na)
        df['Fare'] = df['Fare'].astype(int)

        # Drop unhelpful features

        df = df.drop('Survived', axis=1)
        df = df.drop('PassengerId', axis=1)
        df = df.drop('Ticket', axis=1)

        # Convert category features

        genders = {'male': 0, 'female': 1}
        df['Sex'] = df['Sex'].map(genders)

        ports = {'S': 0, 'C': 1, 'Q': 2}
        df['Embarked'] = df['Embarked'].map(ports)

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
        df['Title'] = df['Title'].fillna(title_na)
        df = df.drop('Name', axis=1)


        df['Age'] = df['Age'].astype(int)
        df.loc[df['Age'] <= 11, 'Age'] = 0
        df.loc[(df['Age'] > 11) & (df['Age'] <= 22), 'Age'] = 1
        df.loc[(df['Age'] > 22) & (df['Age'] <= 33), 'Age'] = 2
        df.loc[(df['Age'] > 33) & (df['Age'] <= 44), 'Age'] = 3
        df.loc[(df['Age'] > 44) & (df['Age'] <= 55), 'Age'] = 4
        df.loc[(df['Age'] > 55) & (df['Age'] <= 66), 'Age'] = 5
        df.loc[df['Age'] > 66, 'Age'] = 6

        ''' Pandas Interval doesn't enumerate propertly
        df['Fare'] = df.Fare.astype(int)
        fare_bands = pd.qcut(df['Fare'], 6)
        for fare_band in fare_bands:
            print(f'left: {fare_band.left} right: {fare_band.right}')
        for i, fare_band in enumerate(fare_bands):
            print(f'i: {i} left: {fare_band.left} right: {fare_band.right}')
            df.loc[(df['Fare'] >= fare_band.left) & (df['Fare'] < fare_band.right), 'Fare'] = i
        '''
        df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
        df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
        df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
        df.loc[(df['Fare'] > 31) & (df['Fare'] <= 99), 'Fare'] = 3
        df.loc[(df['Fare'] > 99) & (df['Fare'] <= 250), 'Fare'] = 4
        df.loc[df['Fare'] > 250, 'Fare'] = 5
        df['Fare'] = df['Fare'].astype(int)

        # Computed features
        if self.data_config.parameters['use_age_times_class']:
            df['Age_Class'] = df['Age'] * df['Pclass']

        if self.data_config.parameters['use_fare_per_person']:
            df['Fare_Per_Person'] = df['Fare'] / (df['relatives'] + 1)
            df['Fare_Per_Person'] = df['Fare_Per_Person'].astype(int)

        return df.values

    def get_dataset_kaggle(self, features, target_feature):
        data_name, data_dir = self.data_config.dataset_name.split('-')
        dir = os.path.abspath(os.path.join(self.Base_Directory, '..'))
        dir = os.path.join(dir, 'kaggle', data_dir)
        file_name = os.path.join(dir, "train.csv")
        df = pd.read_csv(file_name)
        if target_feature:
            y = df[target_feature]
        if data_dir.lower() == 'titanic':
            X = self.preprocess_titanic_data(df)
        elif features:
            X = self.get_features(df, features)
        else:
            X = df.drop(target_feature, 1)

    #y = y.reshape(-1, 1)
        dataset = (X, y)
        return dataset