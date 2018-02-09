import cntk as C


import numpy as np
from keras import models, layers, optimizers

from rembrandtml.core import Score, ScoreType
from rembrandtml.model_implementations.model_impls import MLModelImplementation


class MLModelImplementationCntk(MLModelImplementation):
    def __init__(self, model_config, instrumentation):
        super(MLModelImplementationCntk, self).__init__(model_config, instrumentation)
        self._model = models.Sequential()
        #self._model.add(layers.Dense(18, activation='relu', input_shape=(10000,)))

    def fit(self, X, y, validate=False):
        self._model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_accuracy'])
        self._model.fit(X, y)

    def evaluate(self, X, y):
        score = Score(self.model_config, ScoreType.LOSS, result)
        return score