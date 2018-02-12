import numpy as np
from keras import models, layers, optimizers

from rembrandtml.core import Score, ScoreType
from rembrandtml.model_implementations.model_impls import MLModelImplementation


class MLModelImplementationKeras(MLModelImplementation):
    def __init__(self, model_config, instrumentation):
        super(MLModelImplementationKeras, self).__init__(model_config, instrumentation)
        self._model = models.Sequential()
        #self._model.add(layers.Dense(18, activation='relu', input_shape=(10000,)))

    def fit(self, X, y, validate=False):
        X_train = self.vectorize_sequece(X)
        y_train = np.asanyarray(y).astype('float32')
        '''
        self._model.add(layers.Dense(2056, input_shape=(X.shape[1],), activation='relu'))
        self._model.add(layers.Dropout(0.1))
        self._model.add(layers.Dense(1028, activation='relu'))
        self._model.add(layers.Dropout(0.2))
        self._model.add(layers.Dense(1028, activation='relu'))
        self._model.add(layers.Dropout(0.3))
        self._model.add(layers.Dense(512, activation='relu'))
        self._model.add(layers.Dropout(0.4))
        self._model.add(layers.Dense(1, activation='sigmoid'))
        '''
        self._model.add(layers.Dense(6,input_shape=(X.shape[1],) , activation='relu'))
        self._model.add(layers.Dense(8, activation='relu'))
        self._model.add(layers.Dense(8, activation='relu'))
        self._model.add(layers.Dense(8, activation='relu'))
        self._model.add(layers.Dense(1, activation='sigmoid'))

        self._model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_accuracy'])
        self._model.fit(X, y_train,
                    epochs=100,
                    batch_size=512
                        #,validation_data=(X_val, y_val)
                        )

    def evaluate(self, X, y):
        X_test = self.vectorize_sequece(X)
        y_test = np.asanyarray(y).astype('float32')
        loss, acc, bin_acc = self._model.evaluate(X, y_test)
        score = Score(self.model_config, ScoreType.LOSS, loss)
        score.metrics['accuracy'] = acc
        return score