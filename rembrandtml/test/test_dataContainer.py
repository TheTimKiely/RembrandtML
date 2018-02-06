from unittest import TestCase

from rembrandtml.configuration import DataConfig
from rembrandtml.data import DataContainer
from rembrandtml.test.rml_testing import RmlTest


class TestDataContainer(TestCase, RmlTest):
    def test_build_generator(self):
        self.fail()

    def test_generator(self):
        self.fail()


    def test_prepare_data_sklearn_boston(self):
        data_config = DataConfig('sklearn', 'boston')
        data_container = DataContainer(data_config)
        features = ('RM', 'CRIM')
        data_container.prepare_data(features=features)
        self.assertEqual(len(features), data_container.X.shape[1])

        total_features = 13
        data_container.prepare_data()
        assert_params = self.prepare_assert(total_features, data_container.X.shape[1])
        self.assertEqual(*assert_params)


    def test_prepare_data_sklearn_mnist(self):
        data_config = DataConfig('sklearn', 'mnist-original')
        data_container = DataContainer(data_config)
        data_container.prepare_data()
        X_shape = (70000, 784)
        self.assertEqual(*self.prepare_assert(X_shape, data_container.X.shape))
        data_container.split()
        X_train_shape = (49000, 784)
        self.assertEqual(*self.prepare_assert(X_train_shape, data_container.X_train.shape))


    def test_prepare_file_data(self):
        self.fail()
