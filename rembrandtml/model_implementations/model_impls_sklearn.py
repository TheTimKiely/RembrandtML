import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, \
    ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron

from rembrandtml.core import Score, ScoreType, TuningResults, Prediction, StateError
from rembrandtml.model_implementations.model_impls import MLModelImplementation
from rembrandtml.models import ModelType
from rembrandtml.visualization import Visualizer

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
    def create_ensemble_model_simple(self):
        clf1 = linear_model.LogisticRegression(random_state=1)
        clf2 = RandomForestClassifier(random_state=1)
        clf3 = naive_bayes.GaussianNB()
        model = VotingClassifier(estimators=[
            ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
        return model
    def create_ensemble_model_medium(self):
        from sklearn.svm import SVC
        svc = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
             decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
             max_iter=-1, probability=True, random_state=None, shrinking=True,
             tol=0.001, verbose=False)

        logreg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
                            penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                            verbose=0, warm_start=False)

        rforest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
                                oob_score=False, random_state=None, verbose=0,
                                warm_start=False)

        knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                              metric_params=None, n_jobs=1, n_neighbors=10, p=2,
                              weights='distance')

        xtrees = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                              max_depth=None, max_features='auto', max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=1, min_samples_split=2,
                              min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=1,
                              oob_score=False, random_state=None, verbose=0, warm_start=False)

        gboost= GradientBoostingClassifier(criterion='friedman_mse', init=None,
                                    learning_rate=0.1, loss='deviance', max_depth=3,
                                    max_features='auto', max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=1, min_samples_split=2,
                                    min_weight_fraction_leaf=0.0, n_estimators=30,
                                    presort='auto', random_state=None, subsample=1.0, verbose=0,
                                    warm_start=False)
        return VotingClassifier(estimators=
                                        [('svc', svc), ('logreg',logreg), ('rforst', rforest), ('knn', knn),
                                         ('xtrees', xtrees), ('gboost', gboost)],
                                         voting='hard')

    def create_ensemble_model(self):
        from xgboost import XGBClassifier
        voting_classifier = VotingClassifier(estimators=[
            ('logreg_grid', LogisticRegression(C=100, penalty='l1')),
            ('svc', SVC(kernel = 'rbf', probability=True, random_state = 1, C = 3)),
            ('random_forest', RandomForestClassifier(n_estimators=100,max_depth=9,min_samples_split=6, min_samples_leaf=4)),
            ('gradient_boosting', GradientBoostingClassifier()),
            ('decision_tree', DecisionTreeClassifier( max_depth=5,
                                class_weight = 'balanced',
                                min_weight_fraction_leaf = 0.01)),
            ('decision_tree_grid', DecisionTreeClassifier( max_depth=7,
                                 max_features='auto',
                                class_weight = 'balanced',
                                min_weight_fraction_leaf = 0.01)),
            ('knn_grid', KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', n_jobs = -1)),
            ('XGB Classifier', XGBClassifier()),
            ('BaggingClassifier', BaggingClassifier()),
            ('ExtraTreesClassifier', ExtraTreesClassifier()),
            ('gaussian', GaussianNB())], voting='hard')

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
            self._model = LogisticRegression()
        elif model_config.model_type == ModelType.LINEAR_REGRESSION:
            self._model = LinearRegression()
        elif model_config.model_type == ModelType.SGD_CLASSIFIER:
            self._model = SGDClassifier()
        elif model_config.model_type == ModelType.RANDOM_FOREST_CLASSIFIER:
            self._model = RandomForestClassifier(**self.model_config.parameters)
        elif model_config.model_type == ModelType.DECISTION_TREE_CLASSIFIER:
            from sklearn.tree import DecisionTreeClassifier
            self._model = DecisionTreeClassifier()
        elif model_config.model_type == ModelType.SVC:
            from sklearn.svm import LinearSVC
            self._model = LinearSVC()
        elif model_config.model_type == ModelType.PERCEPTRON:
            self._model = Perceptron()
        elif model_config.model_type == ModelType.NAIVE_BAYES:
            self._model = GaussianNB()
        elif model_config.model_type == ModelType.VOTING_CLASSIFIER:
            # We cannot instantiate the ScikitLearn VotingClassifier until we have instantiated the estimators
            # ToDo move estimator instantiation to ModelFactory
            self._model = self.create_ensemble_model()

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
        if isinstance(self._model, RandomForestClassifier):
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
            values[f'cv:{str(metric)}'] = value
        return values

    def evaluate_metric(self, X, y, metric):
        kfolds = KFold(n_splits=10, random_state=random.randint(1, 100))
        value = cross_val_score(self._model, X, y, cv=kfolds, scoring = metric)
        return (metric, value)


    def evaluate(self, X, y):
        self.validate_trained()
        score_value = self._model.score(X, y)
        values = {str(self.score_type): score_value}
        cv_values = self.evaluate_metrics(X, y, self.metrics)
        values.update(cv_values)
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

        vis = Visualizer()
        vis.plot_model_complexity(neighbors, train_accuracy, test_accuracy)
        vis.show();

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
