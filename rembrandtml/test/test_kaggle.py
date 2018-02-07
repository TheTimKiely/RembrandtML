import os, unittest
import numpy as np
import pandas as pd

from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.test.rml_testing import RmlTest, TestConfig


class KaggleTests(unittest.TestCase, RmlTest):
    def __init__(self):
        RmlTest.__init__(self)

    def log_score(self, score, context_config, file_name):
        with open(file_name, 'a+') as f:
            f.write('++++++++++++++++++++++++++++++++++\n')
            f.write(str(score))
            f.write('\n')
            f.write(f'Model Params: {str(context_config.model_config.parameters)}')
            f.write('\n')
            f.write(f'Data Params: {str(context_config.data_config.parameters)}')
            f.write('\n')

    def run_test(self, data_config, model_config, log_file):
        context_config = ContextConfig(model_config, data_config)
        context = ContextFactory.create(context_config)
        context.prepare_data(target_feature='Survived')
        context.train()
        score = context.evaluate()
        self.results = self.results.append(pd.DataFrame({'Model': [model_config.name], 'Score': [score]}))
        print(score)
        self.log_score(score, context_config, log_file)


    def test_titanic_competition(self):
        log_file = 'scores.txt'
        if os.path.isfile(log_file):
            os.remove(log_file)



        self.test_config = TestConfig('ScikitLearn Logistic Regression', log_file)
        data_config = DataConfig('pandas', 'kaggle-titanic')
        data_config.parameters = {'create_alone_column': True, 'use_age_times_class': True, 'use_fare_per_person': True}
        model_config = ModelConfig(self.test_config.model_name, 'sklearn', ModelType.LOGISTIC_REGRESSION)
        self.run_test(data_config, model_config, log_file)


        model_config.model_type = ModelType.KNN
        model_config.name = 'ScikitLearn KNN'
        self.test_config.model_name = model_config.name
        context_knn = ContextConfig(model_config, data_config)
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.STOCHASTIC_GRAD_DESC_CLASSIFIER
        model_config.name = 'ScikitLearn SGD'
        self.test_config.model_name = model_config.name
        context_knn = ContextConfig(model_config, data_config)
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        model_config.parameters = {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 10,
                                   'n_estimators': 100, 'max_features': 'auto', 'oob_score': True,
                                   'random_state': 1, 'n_jobs':-1}
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

        model_config.model_type = ModelType.STOCHASTIC_GRAD_DESC_CLASSIFIER
        model_config.name = 'ScikitLearn SGD'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)

        model_config.model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        model_config.parameters = {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 10,
                                   'n_estimators': 100, 'max_features': 'auto', 'oob_score': True,
                                   'random_state': 1, 'n_jobs':-1}
        model_config.name = 'ScikitLearn Random Forest'
        self.test_config.model_name = model_config.name
        self.run_test(data_config, model_config, log_file)
        model_config.parameters = {}

        results_sorted = self.results.sort_values(by='Score', ascending=False)
        np.savetxt(r'SortedScored.txt', results_sorted.values, fmt='%s')

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
        #prediction = context.predict(context.model.data_container.X_test)


if __name__ == '__main__':
    unittest.main()
