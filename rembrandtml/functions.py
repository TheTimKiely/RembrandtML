import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    s -- sigmoid(x)
    """

    s = 1 / (1 + np.exp(-x))

    return s

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    Arguments:
    x -- A scalar or numpy array
    Return:
    ds -- Your computed gradient.
    """

    s = sigmoid(x)
    ds = s * (1 - s)

    return ds

def normalize(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    Argument:
    x -- A numpy matrix of shape (n, m)
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """

    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    print(x_norm)
    x = x / x_norm
    ### END

    return x

def logloss(self, predicted, actual, eps=1e-14):
    probability = np.clip(predicted, eps, 1-eps)
    loss = -1 * np.mean(actual * np.log(probability)
                        + (1 - actual)
                        * np.log(1 - predicted))
    return loss


# GRADED FUNCTION: softmax
def softmax(x):
    """Calculates the softmax for each row of the input x.
    Your code should work for a row vector and also for matrices of shape (n, m).
    Argument:
    x -- A numpy matrix of shape (n,m)
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum

    return s

# GRADED FUNCTION: L1
def L1(y, y_pred):
    """
    Arguments:
    y -- vector of size m (true labels)
    y_pred -- vector of size m (predicted labels)
    Returns:
    loss -- the value of the L1 loss function defined above
    """

    loss = np.sum(np.abs(y - y_pred))

    return loss


def L2(y, y_pred):
    """
    Arguments:
    y_pred -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    Returns:
    loss -- the value of the L2 loss function defined above
    """

    loss = np.sum((y - y_pred) ** 2)

    return loss


# GRADED FUNCTION: image2vector
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

    return v