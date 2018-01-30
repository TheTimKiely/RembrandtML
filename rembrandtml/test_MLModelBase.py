from unittest import TestCase

from rembrandtml.configuration import MLConfig, Verbosity
from rembrandtml.data import DataContainer
from rembrandtml.factories import ModelFactory
from rembrandtml.models import ModelType

from sklearn import datasets

class TestMLModelBase(TestCase):
    def __init__(self):
        self.Config = MLConfig(ModelType.LINEAR_REGRESSION, 'sklearn', 't', Verbosity.DEBUG)

    def test_prepare_data(self):
        # Should data be prepared in the model or the DataContainer?
        df = datasets.load_digits()
        data_container = DataContainer()
        model = ModelFactory.create('TestLinearRegression', self.Config)

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = model.prepare_data(df)
        self.assertIsNotNone(model, 'Failed to create MLModel')

    def test_build_model(self):
        self.fail()

    def test_load_from_file(self):
        self.fail()

    def test_fit_and_save(self):
        self.fail()

    def test_evaluate(self):
        self.fail()

    def test_predict(self):
        self.fail()
