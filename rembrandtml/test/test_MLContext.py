import unittest

from rembrandtml.configuration import MLConfig, DataConfig, ModelConfig
from rembrandtml.core import Verbosity
from rembrandtml.entities import MLContext
from rembrandtml.factories import ModelFactory
from rembrandtml.models import ModelType
from rembrandtml.plotting import Plotter


class TestClassifiers(unittest.TestCase):
    def test_knn_sklearn(self):
        # Create DataConfiguration that describes the Dataprovider
        data_config = DataConfig('sklearn', 'mnist-original')

        # Create ModelConfig that describes the model
        model_config = ModelConfig(ModelType.KNN, 'sklearn', )

        # Create ContextConfig that describes context features, such as logging and instrumentation
        config = MLConfig(model_config, data_config, Verbosity.DEBUG)

        # Instantiate the MLContext
        ctxt = MLContext.create(config)

        #Prepare the data
        ctxt.prepare_data()

        #Train the model
        ctxt.train()

        #Evaluate the model
        score = ctxt.evaluate()

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
