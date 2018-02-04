import numpy as np
from sklearn import  linear_model
from sklearn.neighbors import KNeighborsClassifier

from rembrandtml.model_implementations.model_impls import MLModelImplementation
from rembrandtml.plotting import Plotter

class MLModelSkLearn(MLModelImplementation):
    def __init__(self):
        super(MLModelSkLearn, self).__init__()
        self._reg = None

    def fit(self, X, y, validate=False):
        self._reg = linear_model.LinearRegression()
        from sklearn.model_selection import cross_val_score
        self._reg.fit(X, y)

    def evaluate(self, X, y):
        self.validate_trained()
        score = self._reg.score(X, y)
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
        prediction = self._reg.predict(X)
        return prediction