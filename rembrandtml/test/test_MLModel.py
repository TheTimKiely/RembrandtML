from unittest import TestCase
import numpy as np
from rembrandtml.configuration import MLConfig, Verbosity, DataConfig, ModelConfig
from rembrandtml.factories import ModelFactory
from rembrandtml.models import ModelType
from rembrandtml.plotting import Plotter


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
        config.data_config = DataConfig('sklearn', 'boston')
        config.model_config = ModelConfig()
        model = ModelFactory.create('SkLearnLinReg', config)
        features = ('RM')
        model.fit(features=features)

        X_test = model.data_container.X
        y_test = model.data_container.y
        min_x = min(X_test)
        max_x = max(X_test)
        X_test_space = np.linspace(min_x, max_x).reshape(-1, 1)
        prediction = model.predict(X_test_space)
        plotter = Plotter()
        plotter.plot_scatter(X_test, y_test, color='blue')
        plotter.plot(X_test_space, prediction, color='black')
        plotter.show()
        #model.plot()

        features = (['RM', 'CR'])
        model.fit(features)
        prediction = model.predict()
        model.plot()


    def test_load_from_file(self):
        self.fail()

    def test_evaluate(self):
        self.fail()

    def test_predict(self):
        self.fail()
