import numpy as np
import  tensorflow as tf
from rembrandtml.model_implementations.model_impls import MLModelImplementation
from rembrandtml.core import Verbosity

class MLModelTensorflow(MLModelImplementation):


    def fit_normal_equation(self, X, y):
        m, n = X.shape
        housing_data_plus_bias = np.c_[np.ones((m, 1)), X]
        X_train = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
        y_train = tf.constant(y.reshape(-1, 1), dtype=tf.float32, name='y')
        XT_train = tf.transpose(X_train)
        theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT_train, X_train)), XT_train), y_train)
        '''
        theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        lin_reg.fit(housing.data, housing.target.reshape(-1, 1))
        theta_sklearn = np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T]
        '''
        with tf.Session() as sess:
            theta_value = theta.eval()
        return theta_value

    def fit_gradient_descent(self, X, y):
        best_theta = None
        return best_theta

    def fit(self, X, y, validate=False):
        self.log('Fitting with tensorflow', Verbosity.DEBUG)
        theta_value = self.fit_normal_equation(X, y)
        theta_best = self.fit_gradient_descent(X, y)
        self.theta = theta_best

    def predict(self, X):
        self.log('Predicting with tensorflow', Verbosity.DEBUG)
        graph_pred = tf.matmul(X, self.theta)
        with tf.Session() as sess:
            prediction = graph_pred.eval()
        return prediction
