import random

import numpy as np
import  pandas as pd
from sklearn import linear_model, ensemble, model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from rembrandtml.core import Score, ScoreType, TuningResults, Prediction, StateError
from rembrandtml.model_implementations.model_impls import MLModelImplementation
from rembrandtml.models import ModelType
from rembrandtml.plotting import Plotter

class MLModelSkLearn(MLModelImplementation):
    """
    from sklearn.metrics import roc_auc_score
    r_a_score = roc_auc_score(Y_train, y_scores)
    print("ROC-AUC-Score:", r_a_score)

    from sklearn.metrics import roc_curve
    # compute true positive rate and false positive rate
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)

    from sklearn.metrics import precision_recall_curve
    # getting the probabilities of our predictions
    y_scores = random_forest.predict_proba(X_train)
    y_scores = y_scores[:,1]

    precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
    """
    def __init__(self, model_config, instrumentation):
        super(MLModelSkLearn, self).__init__(model_config, instrumentation)
        # Standard metrics for LogisticRegression
        self.metrics = (ScoreType.ACCURACY,)
        # The ScikitLearn metric returned from score() for LogisticRegression
        self.score_type = ScoreType.ACCURACY
        self.score_notes = 'ScikitLearn Implementation: All scorer objects follow the convention that higher return values are better than lower return values.'
        if model_config.model_type == ModelType.KNN:
            self._model = KNeighborsClassifier()
        elif model_config.model_type == ModelType.LOGISTIC_REGRESSION:
            self._model = linear_model.LogisticRegression()
        elif model_config.model_type == ModelType.LINEAR_REGRESSION:
            self._model = linear_model.LinearRegression()
        elif model_config.model_type == ModelType.SGD_CLASSIFIER:
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
        elif model_config.model_type == ModelType.VOTING_CLASSIFIER:
            # We cannot instantiate the ScikitLearn VotingClassifier until we have instantiated the estimators
            pass
        else:
            raise TypeError(f'The model type {self.model_config.model_type} is not suppored for the framework: {self.model_config.framework_name}')


    @property
    def coefficients(self):
        if hasattr(self._model, 'coef_'):
            return self._model.coef_
        else:
            raise StateError('Coefficients are not available because the model has not yet been trained.')

    @property
    def intercepts(self):
        if hasattr(self._model, 'intercept_'):
            return self._model.intercept_
        else:
            raise StateError('Intercepts are not available because the model has not yet been trained.')


    def fit(self, X, y, validate=False):
        self._model.fit(X, y)

    # This class should be RandomForestClassifierImpl to get rid of all these ifs
    def customize_score(self, score):
        if isinstance(self._model, ensemble.RandomForestClassifier):
            #importances = pd.DataFrame(
             #   {'feature': X_train.columns, 'importance': np.round(self._model.feature_importances_, 3)})
            #importances = importances.sort_values('importance', ascending=False).set_index('feature')
            #importances.plot.bar()
            if self._model.oob_score:
                score.values['oob'] = self._model.oob_score_
            score.values['importances'] = self._model.feature_importances_

    def tune(self, X, y, tuning_parameters, model_parameters):
        from sklearn.model_selection import GridSearchCV
        #Check the list of available parameters with `estimator.get_params().keys()`
        keys = self._model.get_params().keys()
        grid = GridSearchCV(self._model, param_grid=model_parameters, **tuning_parameters)
        grid.fit(X, y)
        tuning_results = TuningResults(self.model_config.name, grid.best_params_)
        return tuning_results

    def evaluate_metrics(self, X, y, metrics):
        values = {}
        for metric in metrics:
            name, value = self.evaluate_metric(X, y, str(metric))
            values[metric] = value
        return values

    def evaluate_metric(self, X, y, metric):
        kfolds = model_selection.KFold(n_splits=10, random_state=random.randint(1, 100))
        value = cross_val_score(self._model, X, y, cv=kfolds, scoring = metric)
        return (metric, value)


    def evaluate(self, X, y):
        self.validate_trained()
        values = self.evaluate_metrics(X, y, self.metrics)
        score_value = self._model.score(X, y)
        values[str(self.score_type)] = score_value
        score = Score(self.model_config, values, self.score_notes)
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

    def predict(self, X, with_probabilities):
        self.validate_trained()
        if with_probabilities:
            y_pred = self._model.predict_proba(X)
        else:
            y_pred = self._model.predict(X)
        prediction = Prediction(self.model_config.name, y_pred)
        return prediction

class MLModelSkLearnLinReg(MLModelSkLearn):
    def __init__(self, model_config, instrumentation):
        super(MLModelSkLearnLinReg, self).__init__(model_config, instrumentation)
        self.metrics = (ScoreType.MAE, ScoreType.MSE , ScoreType.R2)
        self.score_type = ScoreType.R2

    def predict(self, X, with_probabilities):
        self.validate_trained()
        # SkLearn LinearRegression does not support propbabilities
        y_pred = self._model.predict(X)
        prediction = Prediction(self.model_config.name, y_pred)
        return prediction