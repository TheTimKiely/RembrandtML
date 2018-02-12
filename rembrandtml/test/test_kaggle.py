import os, unittest
import numpy as np
import pandas as pd

from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig, RunConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.test.rml_testing import RmlTest


class KaggleTests(unittest.TestCase, RmlTest):
    def __init__(self):
        RmlTest.__init__(self)

    def log_score(self, score, context_config, features, file_name):
        with open(file_name, 'a+') as f:
            f.write('++++++++++++++++++++++++++++++++++\n')
            f.write(str(score))
            f.write('\n')
            f.write(f'Model Params: {str(context_config.model_config.parameters)}')
            f.write('\n')
            f.write(f'Data Params: {features}')
            f.write('\n')

    def create_submission(self, index_name, index_values, prediction, file_name):
        submission = pd.DataFrame({
            self.run_config.index_name: index_values,
            self.run_config.prediction_column: prediction
        })
        submission.to_csv(file_name, index=False)

    def run_test(self, data_config, model_config, log_file = None, submit = False):
        context_config = ContextConfig(model_config)
        context = ContextFactory.create(context_config)
        features = ('Has_Cabin', 'Deck', 'Family_Size', 'Alone', 'Port', 'Pclass',
                    'Sex', 'Age_Class', 'SibSp', 'Parch', 'Fare', 'Embarked')
        features = ('Has_Cabin', 'Deck', 'Family_Size', 'Port', 'Sex', 'Age_Class', 'SibSp', 'Fare', 'Embarked')
        #features = ('Deck', 'Family_Size', 'Pclass', 'Sex', 'Age_Class')
        context.prepare_data(features = features, target_feature='Survived')
        context.train()
        score = context.evaluate()
        self.results = self.results.append(pd.DataFrame({'Model': [model_config.name], 'Score': [score],
                                                         'Model Parameters': [model_config.parameters],
                                                         'Data Features': [features]}))

        if 'importances' in score.metrics.keys():
            df = pd.DataFrame({'feature': context.model.data_container.X_columns, 'importance': np.round(score.metrics['importances'], 3)})
            importances = df.sort_values('importance', ascending=False).set_index('feature')
            importances.plot.bar()

        if log_file:
            self.log_score(score, context_config, features, log_file)

        if submit:
            prediction_file = self.get_data_file_path('kaggle', data_config.dataset_name, 'test.csv')
            X_pred = context.model.data_container.get_prediction_data(features, prediction_file)
            prediction = context.predict(X_pred)
            index_values = context.model.data_container.get_column_values(prediction_file, self.run_config.index_name)
            self.create_submission(self.run_config.prediction_index,
                                   index_values, prediction.values, f'submission_{model_config.name}.csv')


    def test_tune_titanic_competition(self):
        self.run_config = RunConfig('ScikitLearn Random Forest')
        dataset_name = 'titanic'
        dataset_file_path = self.get_data_file_path('kaggle', dataset_name, 'train.csv')
        data_config = DataConfig('pandas', dataset_name, dataset_file_path)
        data_config.parameters = {'create_alone_column': True, 'use_age_times_class': True, 'use_fare_per_person': True}
        model_config = ModelConfig(self.run_config.model_name, 'sklearn', ModelType.RANDOM_FOREST_CLASSIFIER, data_config)
        model_config.model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        model_parameters = {'criterion': ('gini', 'entropy'), 'min_samples_leaf': range(1, 5), 'min_samples_split': range(2, 15),
                                   'n_estimators': (2, 5, 40, 200), 'max_features': (2, 3, None, 'auto'),
                                   'class_weight': (None, 'balanced', 'balanced_subsample')}
        model_parameters = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
        #model_parameters = {'n_estimators': (5, 10, 20, 50, 100, 200), 'max_features': (2, 3, None, 'auto'), 'class_weight': (None, 'balanced', 'balanced_subsample')}
        tuning_parameters = {'cv': 3}
        context = ContextFactory.create(ContextConfig(model_config, data_config))
        context.prepare_data(target_feature='Survived')
        tuning_results = context.tune(tuning_parameters, model_parameters)
        print(str(tuning_results.best_params))


    def test_titanic_competition(self):

        log_file = 'scores.txt'
        if os.path.isfile(log_file):
            os.remove(log_file)

        dataset_name = 'titanic'
        self.run_config = RunConfig('ScikitLearn Logistic Regression', log_file)
        self.run_config.prediction_column = 'Survived'
        self.run_config.prediction_index = 1
        self.run_config.index_name = 'PassengerId'


        dataset_file_path = self.get_data_file_path('kaggle', dataset_name, 'train.csv')
        data_config = DataConfig('pandas', dataset_name, dataset_file_path)
        model_config = ModelConfig(self.run_config.model_name, 'sklearn', ModelType.LOGISTIC_REGRESSION, data_config)
        #self.run_test(data_config, model_config, log_file, True)

        model_config.model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        model_config.parameters = {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 10,
                                   'n_estimators': 100, 'max_features': 'auto', 'oob_score': True,
                                   'random_state': 1, 'n_jobs':-1}
        model_config.parameters = {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 1500, 'oob_score': True}
        model_config.name = 'ScikitLearn Random Forest'
        self.run_config.model_name = model_config.name
        self.run_test(data_config, model_config, submit=True)
        model_config.parameters = {}

        model_config.model_type = ModelType.VOTING_CLASSIFIER
        model_config.name = 'SkLearn Voting'
        self.run_test(data_config, model_config)

        '''
        #model_config.model_type = ModelType.STACKED
        #model_config.name = 'SkLearn Stacked'
        #self.run_test(data_config, model_config)



        #model_config.framework_name = 'cntk'
        #model_config.model_type = ModelType.LOGISTIC_REGRESSION
        #model_config.name = 'CNTK LogReg'
        #self.test_config.model_name = model_config.name
        #context_knn = ContextConfig(model_config, data_config)
        #self.run_test(data_config, model_config, log_file)

        
        #model_config.framework_name = 'keras'
        #model_config.model_type = ModelType.LOGISTIC_REGRESSION
        #model_config.name = 'Keras RNN'
        #self.test_config.model_name = model_config.name
        #context_knn = ContextConfig(model_config, data_config)
        #self.run_test(data_config, model_config, log_file)

        
        model_config.model_type = ModelType.KNN
        model_config.name = 'ScikitLearn KNN'
        self.test_config.model_name = model_config.name
        context_knn = ContextConfig(model_config, data_config)
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.SGD_CLASSIFIER
        model_config.name = 'ScikitLearn SGD'
        self.test_config.model_name = model_config.name
        context_knn = ContextConfig(model_config, data_config)
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        model_config.parameters = {'criterion': 'gini', 'min_samples_leaf': 4, 'min_samples_split': 12,
                                   'n_estimators': 5, 'max_features': 'auto', 'oob_score': True,
                                   'random_state': 1, 'n_jobs':-1,
                                   'class_weight': None}
        model_config.name = 'ScikitLearn Random Forest'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)
        model_config.parameters = {}

        model_config.model_type = ModelType.LOGISTIC_REGRESSION
        data_config.parameters = {'create_alone_column': False, 'use_age_times_class': False, 'use_fare_per_person': False}
        model_config.name = 'ScikitLearn KNN'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.KNN
        model_config.name = 'ScikitLearn KNN'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.SGD_CLASSIFIER
        model_config.name = 'ScikitLearn SGD'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)
        '''

        self.results = self.results.sort_values(by='Score', ascending=False)
        np.savetxt(log_file, self.results.values, fmt='%s')

        print('done')
        '''
        context = ContextFactory.create(ContextConfig)
        context.prepare_data(target_feature='Survived')
        context.train()
        score = context.evaluate()
        print(score)
        with open('scores.txt', 'a+') as f:
            f.write('++++++++++++++++++++++++++++++++++\n')
            f.write(str(score))
            f.write('\n')
            f.write(f'Data Params: {str(data_config.parameters)}')
            f.write('\n')
        '''

if __name__ == '__main__':
    unittest.main()
