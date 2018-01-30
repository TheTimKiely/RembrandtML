from unittest import TestCase

from rembrandtml.configuration import MLConfig, Verbosity, DataConfig, ModelConfig
from rembrandtml.models import ModelType


class TestMLModel(TestCase):
    def test_build_model(self):
        self.fail()

    def test_train(self):
        self.fail()

    def test_fit_classification(self):
        # ToDo: Should there be seperate model classes for classification and linear regression
        config = MLConfig(ModelType.SIMPLE_CLASSIFICATION, 'sklearn', 't', Verbosity.DEBUG)

    def test_fit_linear_regression(self):
        config = MLConfig(ModelType.LINEAR_REGRESSION, 'sklearn', 't', Verbosity.DEBUG)
        config.data_config = DataConfig()
        config.model_config = ModelConfig()

    def test_load_from_file(self):
        self.fail()

    def test_evaluate(self):
        self.fail()

    def test_predict(self):
        self.fail()
