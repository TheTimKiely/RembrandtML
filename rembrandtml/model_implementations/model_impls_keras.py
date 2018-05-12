import os
import numpy as np
from keras import models, layers, optimizers
from rembrandtml.configuration import Verbosity

from rembrandtml.core import Score, ScoreType, Prediction
from rembrandtml.model_implementations.model_impls import MLModelImplementation


class MLModelImplementationKeras(MLModelImplementation):
    def __init__(self, model_config, instrumentation):
        super(MLModelImplementationKeras, self).__init__(model_config, instrumentation)
        self._model = models.Sequential()
        #self._model.add(layers.Dense(18, activation='relu', input_shape=(10000,)))

    def fit(self, X, y, validate=False):
        #X_train = self.vectorize_sequece(X)
        #y_train = np.asanyarray(y).astype('float32')
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
        self._model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.MaxPooling2D((2,2)))
        self._model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self._model.add(layers.MaxPooling2D((2,2)))
        self._model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        self._model.add(layers.MaxPooling2D((2,2)))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dropout(0.5))
        self._model.add(layers.Dense(512, activation='relu'))
        self._model.add(layers.Dense(1, activation='sigmoid'))

        self._model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_accuracy'])
        history = self._model.fit(X, y,
                    epochs=self.model_config.epochs,
                    batch_size=512
                        #,validation_data=(X_val, y_val)
                        )
        acc_score = history.history['acc'][len(history.history['acc']) - 1]
        self.log(f'Accuracy: {acc_score}')

    def save(self, model_path, model_arch_path, weights_path):
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            self.log(f'Model path directory, {model_dir}, doesn\'t exist.  Creating it.', verbosity=Verbosity.QUIET)
            os.mkdir(model_dir)

        self.log(f'Saving model to: {model_path}', verbosity=Verbosity.QUIET)
        self._model.save(model_path)

        self.log(f'Saving model architecture to: {model_arch_path}', verbosity=Verbosity.QUIET)
        with open(model_path, 'w') as fh:
            fh.write(self._model.to_json())
        self.log(f'Saved model to: {model_path}', verbosity=Verbosity.QUIET)

        self.log(f'Saving weights to: {weights_path}', verbosity=Verbosity.QUIET)
        self._model.save_weights(weights_path)
        self.log(f'Saved model to: {weights_path}', verbosity=Verbosity.QUIET)

    def evaluate(self, X, y):
        X_test = self.vectorize_sequece(X)
        y_test = np.asanyarray(y).astype('float32')
        loss, acc, bin_acc = self._model.evaluate(X, y_test)
        score = Score(self.model_config, ScoreType.LOSS, loss)
        score.metrics['accuracy'] = acc
        return score

    def predict(self, X, with_probability):
        y_pred = self._model.predict(X)
        y_pred_classes = self._model.predict_classes(X)
        prediction = Prediction(self.model_config.name, y_pred)
        return prediction
