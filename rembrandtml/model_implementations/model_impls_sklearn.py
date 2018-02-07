import numpy as np
from sklearn import  linear_model, ensemble
from sklearn.neighbors import KNeighborsClassifier

from rembrandtml.core import Score, ScoreType
from rembrandtml.model_implementations.model_impls import MLModelImplementation
from rembrandtml.models import ModelType
from rembrandtml.plotting import Plotter

class MLModelSkLearn(MLModelImplementation):
    def __init__(self, model_config, instrumentation):
        super(MLModelSkLearn, self).__init__(model_config, instrumentation)
        if model_config.model_type == ModelType.KNN:
            self._model = KNeighborsClassifier()
        elif model_config.model_type == ModelType.LOGISTIC_REGRESSION:
            self._model = linear_model.LogisticRegression()
        elif model_config.model_type == ModelType.LINEAR_REGRESSION:
            self._model = linear_model.LinearRegression()
        elif model_config.model_type == ModelType.STOCHASTIC_GRAD_DESC_CLASSIFIER:
            self._model = linear_model.SGDClassifier()
        elif model_config.model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
            self._model = ensemble.RandomForestClassifier(**self.model_config.parameters)
        elif model_config.model_type == ModelType.DECISTION_TREE_CLASSIFIER:
            from sklearn.tree import DecisionTreeClassifier
            self._model = DecisionTreeClassifier()
        elif model_config.model_type == ModelType.SVC:
            from sklearn.svm import LinearSVC
            self._model = LinearSVC()
        elif model_config.model_type == ModelType.PERCEPTRON:
            self._model = linear_model.Perceptron()
        elif model_config.model_type == ModelType.NAIVE_BAYES:
            from sklearn.naive_bayes import GaussianNB
            self._model = GaussianNB()
        else:
            raise TypeError(f'The model type {self.model_config.model_type} is not suppored for the framework: {self.model_config.framework_name}')


    def fit(self, X, y, validate=False):
        self._model.fit(X, y)

    # This class should be RandomForestClassifierImpl to get rid of all these ifs
    def customize_score(self, score):
        if isinstance(self._model, ensemble.RandomForestClassifier):
            if self._model.oob_score:
                score.metrics.append('oob', self._model.oob_score_)


    def evaluate(self, X, y):
        self.validate_trained()
        value = self._model.score(X, y)
        score = Score(self.model_config, ScoreType.R2, value)
        self.customize_score(score)
        return score

    def train(self):
        # Setup arrays to store train and test accuracies
        neighbors = np.arange(1, 9)
        train_accuracy = np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))

        # Loop over different values of k
        for i, k in enumerate(neighbors):
            # Setup a k-NN Classifier with k neighbors: knn
            knn = KNeighborsClassifier(n_neighbors=k)

            # Fit the classifier to the training data
            knn.fit(self.data_container.X_train, self.data_container.y_train)

            # Compute accuracy on the training set
            train_accuracy[i] = knn.score(self.data_container.X_train, self.data_container.y_train)

            # Compute accuracy on the testing set
            test_accuracy[i] = knn.score(self.data_container.X_test, self.data_container.y_test)

        plotter = Plotter()
        plotter.plot_model_complexity(neighbors, train_accuracy, test_accuracy)
        plotter.show();

    def predict(self, X):
        self.validate_trained()
        prediction = self._model.predict(X)
        return prediction