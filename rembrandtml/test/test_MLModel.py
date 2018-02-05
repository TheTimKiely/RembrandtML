from unittest import TestCase
import numpy as np
from rembrandtml.configuration import ContextConfig, DataConfig, ModelConfig, Verbosity
from rembrandtml.factories import ModelFactory
from rembrandtml.models import ModelType
from rembrandtml.plotting import Plotter

# ToDo make data-driven test
# @ddt
class TestMLModel(TestCase):

    def setUpModule():
        print("setup module")

    def tearDownModule():
        print("teardown module")

        @classmethod
        def setUpClass(self):
            print("foo setUpClass")

        @classmethod
        def tearDownClass(self):
            print("foo tearDownClass")

        def setUp(self):
            print("foo setUp")

        def tearDown(self):
            print("foo tearDown")

    def test_build_model(self):
        self.fail()

    def test_train(self):
        self.fail()

    def test_fit_classification(self):
        # ToDo: Should there be seperate model classes for classification and linear regression
        config = ContextConfig(ModelType.SIMPLE_CLASSIFICATION, 'sklearn', 't', Verbosity.DEBUG)

    # ToDo make data-driven test
    # @data('boston', 'ca_housing')
    def test_fit_linear_regression_tensorflow(self):
        config = ContextConfig(ModelType.LINEAR_REGRESSION, 'tensorflow', 't', Verbosity.DEBUG)
        config.data_config = DataConfig('sklearn', 'ca_housing')
        config.model_config = ModelConfig()
        model = ModelFactory.create('TensorFlowLinReg', config)
        features = ('AveBedrms')
        #model.fit(features=features)
        model.fit()
        X_test = model.data_container.X_test
        y_test = model.data_container.y_test
        prediction = model.predict(model.data_container.X_test)
        plotter = Plotter()
        plotter.plot_scatter(X_test, y_test, color='blue')
        plotter.plot(X_test, prediction, color='black')
        plotter.show()
        #model.plot()

        features = (['AveBedrms', 'Population'])
        model.fit(features=features)

    def test_fit_linear_regression_sklearn(self, framework='sklearn', dataset='boston', features=('RM', 'CRIM')):
        config = ContextConfig(ModelType.LINEAR_REGRESSION, 'sklearn', 't', Verbosity.DEBUG)

        '''
        from sklearn import datasets
        
        boston = datasets.load_boston()
        X = boston['data']
        y = boston['target']
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        reg = LinearRegression()
        scores = cross_val_score(reg, X, y, cv=6)
        p = reg.predict(X)
        '''

        config.data_config = DataConfig(framework, dataset)
        config.model_config = ModelConfig()
        model = ModelFactory.create('SkLearnLinReg', config)
        '''
        features = ('RM')
        model.fit(features=features)
        X_test = model.data_container.X_test
        y_test = model.data_container.y_test
        score = model.evaluate(X_test, y_test)
        min_x = min(X_test)
        max_x = max(X_test)
        X_test_space = np.linspace(min_x, max_x).reshape(-1, 1)
        prediction = model.predict(X_test_space)
        plotter = Plotter()
        plotter.plot_scatter(X_test, y_test, color='blue')
        plotter.plot(X_test_space, prediction, color='black')
        plotter.show()
        # model.plot()

        features = (['RM', 'CRIM'])
        '''
        model.fit(features=features)

        X_test = model.data_container.X_test
        y_test = model.data_container.y_test
        score = model.evaluate(X_test, y_test)
        print(f'score: {score}')
        prediction = model.predict(X_test)
        plotter = Plotter()
        plotter.clear()
        X_test = model.data_container.X
        y_test = model.data_container.y

    def test_fit_linear_regression_sklearn_single_feature(self):
        config = ContextConfig(ModelType.LINEAR_REGRESSION, 'sklearn', 't', Verbosity.DEBUG)
        config.data_config = DataConfig('sklearn', 'boston')
        config.model_config = ModelConfig()
        model = ModelFactory.create('SkLearnLinReg', config)
        features = ('RM')
        model.fit(features=features)
        X_test = model.data_container.X_test
        y_test = model.data_container.y_test
        score = model.evaluate(X_test, y_test)
        min_x = min(X_test)
        max_x = max(X_test)
        X_test_space = np.linspace(min_x, max_x).reshape(-1, 1)
        prediction = model.predict(X_test_space)
        plotter = Plotter()
        plotter.plot_scatter(X_test, y_test, color='blue')
        plotter.plot(X_test_space, prediction, color='black')
        plotter.show()
        #model.plot()

        features = (['RM', 'CRIM'])
        model.fit(features=features)

        X_test = model.data_container.X_test
        y_test = model.data_container.y_test
        prediction_2_features = model.predict(X_test)
        plotter.clear()
        X_test = model.data_container.X
        y_test = model.data_container.y
        return
        # ToDo figure out how to plot 3D
        plotter.plot_scatter(X_test[:,0],X_test[:,1], y_test, color='blue')
        plotter.plot(X_test_space, prediction, color='black')
        plotter.show()

        #model.plot()


    def test_load_from_file(self):
        self.fail()

    def test_evaluate(self):
        self.fail()

    def test_predict(self):
        self.fail()
