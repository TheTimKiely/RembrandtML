import unittest

from rembrandtml.configuration import ContextConfig, DataConfig, ModelConfig, Verbosity
from rembrandtml.core import MLContext
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.test.rml_testing import RmlTest


class TestClassifiers(unittest.TestCase, RmlTest):
    def __init__(self):
        super(TestClassifiers, self).__init__()

    def test_cntk(self):
        #data_file = self.get_data_file_path()
        data_file = 'D:\code\ML\data\msdn\wheat\seeds_dataset.txt'
        data_config = DataConfig('pandas', 'wheat', data_file)
        data_config.file_separator = '\t'
        modle_config = ModelConfig('CNTK SVM', 'cntk', ModelType.SGD_CLASSIFIER)
        context = ContextFactory.create(ContextConfig(modle_config, data_config))

        context.prepare_data(target_feature='type')
        context.fit()


    def test_knn_sklearn(self):
        # Create DataConfiguration that describes the Dataprovider
        data_config = DataConfig('sklearn', 'mnist-original')

        # Create ModelConfig that describes the model
        model_config = ModelConfig(ModelType.KNN, 'sklearn')

        # Create ContextConfig that describes context features, such as logging and instrumentation
        config = ContextConfig(model_config, data_config, Verbosity.DEBUG)

        # Instantiate the MLContext
        ctxt = ContextFactory.create(config)

        #Prepare the data
        ctxt.prepare_data()

        #Train the model
        ctxt.train()

        #Evaluate the model
        score = ctxt.evaluate()
        print(f'Model score: {str(score)}')

    def test_knn_sklearn_tune(self):
        # Create DataConfiguration that describes the Dataprovider
        data_config = DataConfig('sklearn', 'mnist-original')

        # Create ModelConfig that describes the model
        model_config = ModelConfig(ModelType.KNN, 'sklearn')

        # Create ContextConfig that describes context features, such as logging and instrumentation
        config = ContextConfig(model_config, data_config, Verbosity.DEBUG)

        # Instantiate the MLContext
        ctxt = ContextFactory.create(config)

        # Prepare the data
        ctxt.prepare_data()

        # Train the model
        ctxt.train()

        # Evaluate the model
        score = ctxt.evaluate()
        print(f'Model score: {str(score)}')

    def test_knn(self):
        import numpy as np
        from sklearn import datasets
        from sklearn.model_selection import  train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        mnist = datasets.fetch_mldata('MNIST original')
        X,y = mnist['data'], mnist['target']
        print(X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)
        y_train_large = (y_train >= 7)
        y_train_odd = (y_train % 2 == 1)
        y_multilabel = np.c_[y_train_large, y_train_odd]
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(X_train, y_train)
        acc = knn.score(X_test, y_test)
        print(acc)

if __name__ == '__main__':
    unittest.main()
