import numpy as np


def logloss(self, predicted, actual, eps=1e-14):
    probability = np.clip(predicted, eps, 1-eps)
    loss = -1 * np.mean(actual * np.log(probability)
                        + (1 - actual)
                        * np.log(1 - predicted))
    return loss