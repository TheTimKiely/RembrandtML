import os, sys
root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
print(root_dir)
sys.path.append(root_dir)
print(f'path: {sys.path}')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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
            self.score.values[ScoreType.MSE] = mse.eval()

        return best_theta

    def fit(self, X, y, validate=False):
        #self.fit_normal_equation(X, y)
        self.log('Fitting with tensorflow using gradient descent.', Verbosity.DEBUG)
        self.theta = self.fit_gradient_descent(X, y)
        #self.theta = self.fit_gradient_autodif(X, y)

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

class MLModelTensorflowCNN(MLModelTensorflow):
    def __init__(self, model_config, instrumentation):
        super(MLModelTensorflowCNN, self).__init__(model_config, instrumentation)

    def fit(self, X, y, validate=False):
        pass

    def build_model(self, X, mode):
        input_layer = tf.reshape(X, [-1, 28, 28, 1])
        '''
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=32,
                                 kernel_size=[5,5],
                                 padding='same',
                                 activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2,2],
                                        strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 padding='same',
                                 activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat,
                                units=1024,
                                activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense,
                                    rate=0.4,
                                    training=tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {'classes': tf.argmax(input=logits, axis=1),
                       'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=y, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        tensors_to_log = {"probabilities": "softmax_tensor"}
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
        '''

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    tf.logging.set_verbosity(tf.logging.INFO)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
    mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()

