import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.tests.test_kernel_approximation import rng

from rembrandtml.core import Score, ScoreType, Prediction
from rembrandtml.model_implementations.model_impls import MLModelImplementation
from rembrandtml.configuration import Verbosity


class MLModelTensorflow(MLModelImplementation):
    def __init__(self, model_config, instrumentation):
        super(MLModelTensorflow, self).__init__(model_config, instrumentation)
        self.epochs = 1000
        self.learning_rate = 0.01
        self._score = None

    @property
    def score(self):
        if self._score is None:
            self._score = Score(self.model_config, notes=self.score_notes)
        return self._score

    def fit_normal_equation(self, X_train, y_train):
        m, n = X_train.shape
        data_plus_bias = np.c_[np.ones((m, 1)), X_train]
        X = tf.constant(data_plus_bias, dtype=tf.float32, name='X')
        y = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32, name='y')
        XT = tf.transpose(X)
        theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
        '''
        theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        lin_reg.fit(housing.data, housing.target.reshape(-1, 1))
        theta_sklearn = np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T]
        '''
        with tf.Session() as sess:
            theta_value = theta.eval()
        return theta_value

    def fit_gradient_optimizer(self, X_train, y_train):
        m, n = X_train.shape
        y_train = y_train.reshape(-1, 1)
        '''
        train_X = np.asarray([[3.3, 4.6], [4.4, 4.6], [5.5, 4.6], [6.71, 4.6], [6.93, 4.6], [4.168, 4.6], [9.779, 4.6],
        [6.182, 4.6], [7.59, 4.6], [2.167, 4.6],[7.042, 4.6], [10.791, 4.6], [5.313, 4.6], [7.997, 4.6], [5.654, 4.6],
                              [9.27, 4.6], [3.1, 4.6]])
        train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                                 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

        X_train = np.asarray(train_X)
        y_train = np.asarray(train_Y).reshape(-1, 1)
        m = train_X.shape[0]
        '''
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        W = tf.Variable(rng.rand(), name='weights')
        b = tf.Variable(rng.rand(), name='bias')

        xs = np.linspace(-3, 3, 55)
        pred = tf.add(tf.multiply(X, W), b)
        cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * m)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                for (x, y) in zip(X_train, y_train):
                    sess.run(optimizer, feed_dict={X: x, Y: y})
                if epoch % 100 == 0:
                    c = sess.run(cost, feed_dict={X: x, Y: y})
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                          "W=", sess.run(W), "b=", sess.run(b))

            train_cost = sess.run(cost, feed_dict={X: X_train, Y: y_train})
            print("Training cost=", train_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
            weights = sess.run(W)
            y_pred = weights * X_train + sess.run(b)

        plt.plot(X_train, y_train, 'ro', label='Training data')
        #plt.plot(X_train, y_pred, label='Fitted regression')
        plt.show()

        return weights

    def fit_gradient_autodif(self, X_train, y_train):
        m, n = X_train.shape
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_train)
        scaled_data_plus_bias = np.c_[np.ones((m, 1)), scaled_data]
        X = tf.constant(scaled_data_plus_bias, dtype=tf.float32, name="X")
        y = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32, name="y")
        theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
        y_pred = tf.matmul(X, theta, name="predictions")
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name="mse")

        gradients = tf.gradients(mse, [theta])[0]

        training_op = tf.assign(theta, theta - self.learning_rate * gradients)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.epochs):
                if epoch % 100 == 0:
                    print("Epoch", epoch, "MSE =", mse.eval())
                sess.run(training_op)

            best_theta = theta.eval()
            predictions = y_pred.eval()

            self.score.values[ScoreType.MSE] = mse.eval()

        return best_theta

    def fit_gradient_descent(self, X_train, y_train):
        # normalize input data
        m, n = X_train.shape
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_train)
        scaled_data_plus_bias = np.c_[np.ones((m, 1)), scaled_data]
        # Set model weights
        W = tf.Variable(rng.randn(), name="weight")
        b = tf.Variable(rng.randn(), name="bias")
        X = tf.constant(scaled_data_plus_bias, dtype=tf.float32, name='X')
        y = tf.constant(y_train.reshape(-1, 1), dtype=tf.float32, name='y')
        theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name='theta')
        y_pred = tf.matmul(X, theta, name='predictions')
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name='mse')
        gradients = 2/m * tf.matmul(tf.transpose(X), error)
        training_op = tf.assign(theta, theta - self.learning_rate * gradients)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.epochs):
                if epoch % 100 == 0:
                    print("Epoch", epoch, "MSE =", mse.eval())
                    self.score.values[ScoreType.MSE] = mse.eval()
                sess.run(training_op)

            best_theta = theta.eval()

        return best_theta

    def fit(self, X, y, validate=False):
        #self.fit_normal_equation(X, y)
        self.log('Fitting with tensorflow using gradient descent.', Verbosity.DEBUG)
        #self.theta = self.fit_gradient_descent(X, y)
        self.theta = self.fit_gradient_autodif(X, y)

    def fit_normal_eq(self, X, y, validate=False):
        m, n = X.shape
        data_plus_bias = np.c_[np.ones((m, 1)), X]
        self.log('Fitting with tensorflow using the normal equation.', Verbosity.DEBUG)
        self.theta = self.fit_normal_equation(data_plus_bias, y)

    def evaluate(self, X, y):
        return self.score

    def predict(self, X_pred, with_probabilities):
        self.log('Predicting with tensorflow', Verbosity.DEBUG)
        m, n = X_pred.shape
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_pred)
        scaled_data_plus_bias = np.c_[np.ones((m, 1)), scaled_data]
        X = tf.constant(scaled_data_plus_bias, dtype=tf.float32, name='X')
        #graph_pred = tf.matmul(X, self.theta)
        y_pred = tf.matmul(X, self.theta, name="predictions")

        with tf.Session() as sess:
            prediction = y_pred.eval()
        return Prediction(self.model_config.name, prediction)

    @property
    def coefficients(self):
        #model_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        return self.theta
