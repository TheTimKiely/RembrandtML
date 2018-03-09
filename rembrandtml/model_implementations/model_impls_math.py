import numpy as np

from rembrandtml.model_implementations.model_impls import MLModelImplementation


class MLModelMath(MLModelImplementation):

    def __init__(self, model_config, instrumentation):
        super(MLModelMath, self).__init__(model_config, instrumentation)
        self.w = None
        self.b = None

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """

        w = np.zeros(shape=(dim, 1), dtype=np.float32)
        b = 0

        assert (w.shape == (dim, 1))
        assert (isinstance(b, float) or isinstance(b, int))

        return w, b

    def sigmoid(self, x):
        """
        Compute the sigmoid of x
        Arguments:
        x -- A scalar or numpy array of any size
        Return:
        s -- sigmoid(x)
        """

        s = 1 / (1 + np.exp(-x))

        return s

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """

        m = X.shape[1]

        A = self.sigmoid((w.T.dot(X)) + b)  # compute activation
        cost = -(np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A))) / m  # compute cost

        dw = (np.dot(X, (A - Y).T)) / m
        db = np.sum(A - Y) / m

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=True):
        """
        This function optimizes w and b by running a gradient descent algorithm
        theta = theta - (alpha * d_theta)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """

        costs = []

        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)

            dw = grads["dw"]
            db = grads["db"]

            w = w - (learning_rate * dw)
            b = b - (learning_rate * db)

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost and i % 10 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs

    def fit(self, X, y, validate=False):

        X = X.T
        y = y.reshape((1, y.shape[0]))

        w, b = self.initialize_with_zeros(X.shape[0])
        parameters, grads, costs = self.optimize(w, b, X, y,
                                                 self.model_config.epochs,
                                                 self.model_config.learning_rate)
        # Retrieve parameters w and b from dictionary "parameters"
        self.w = parameters["w"]
        self.b = parameters["b"]

    def predict(self, X, with_probabilities):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        X = X.T
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = self.w.reshape(X.shape[0], 1)

        A = self.sigmoid(self.w.T.dot(X) + self.b)

        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            Y_prediction[0, i] = 1 if A[0, i] >= 0.5 else 0

        assert (Y_prediction.shape == (1, m))

        return Y_prediction
