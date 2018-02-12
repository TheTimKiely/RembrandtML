import os
from unittest import TestCase
import numpy as np
from rembrandtml.configuration import RunConfig, ContextConfig, DataConfig, ModelConfig, Verbosity, EnsembleModelConfig, \
    EnsembleConfig
from rembrandtml.factories import ModelFactory, ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.plotting import Plotter
from rembrandtml.test.rml_testing import RmlTest


class TestEnsembleModels(TestCase, RmlTest):
    def __init__(self):
        self.run_config = RunConfig('ScikitLearn Ensemble - Voting',)
        self.run_config.prediction_column = 'Survived'
        self.run_config.prediction_index = 1
        self.run_config.index_name = 'PassengerId'
        self.run_config.dataset_name = 'titanic'
        self.run_config.dataset_file_path = self.get_data_file_path('kaggle', self.run_config.dataset_name, 'train.csv')
        self.run_config.log_file = 'scores.txt'

    def test_voting_sklearn_estimators_error(self):
        try:
            data_config = self.init_data_config()
            data_config = DataConfig('pandas', self.run_config.dataset_name, self.run_config.dataset_file_path)

            ensemble_config = EnsembleConfig(estimator_configs)
            model_config = EnsembleModelConfig(self.run_config.model_name, 'sklearn', ModelType.VOTING_CLASSIFIER, data_config, ensemble_config)
            context_config = ContextConfig(model_config, data_config)
            context = ContextFactory.create(context_config)
        except TypeError as err:
            error_string = 'All ensemble models must be configured with estimators'
            if error_string not in err.args[0]:
                self.fail(f'Didn\'t find expected error: {error_string}')

    def test_voting_sklearn(self):
        if os.path.isfile(self.run_config.log_file):
            os.remove(self.run_config.log_file)


        model_config = ModelConfig(self.run_config.model_name, 'sklearn', ModelType.LOGISTIC_REGRESSION)
        # self.run_test(data_config, model_config, log_file, True)

        model_config.model_type = ModelType.RANDOM_FOREST_CLASSIFIER
        model_config.parameters = {'criterion': 'gini', 'min_samples_leaf': 1, 'min_samples_split': 10,
                                   'n_estimators': 100, 'max_features': 'auto', 'oob_score': True,
                                   'random_state': 1, 'n_jobs': -1}
        model_config.parameters = {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 10,
                                   'n_estimators': 1500, 'oob_score': True}
        model_config.name = 'ScikitLearn Random Forest'
        self.run_config.model_name = model_config.name
        self.run_test(data_config, model_config, submit=True)
        model_config.parameters = {}

        model_config.model_type = ModelType.VOTING_CLASSIFIER
        model_config.name = 'SkLearn Voting'
        self.run_test(data_config, model_config)

