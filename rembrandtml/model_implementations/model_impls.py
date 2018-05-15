import numpy as np

from rembrandtml.entities import MLEntityBase


class MLModelImplementation(MLEntityBase):
    def __init__(self, model_config, instrumentation):
        super(MLModelImplementation, self).__init__(instrumentation)
        self._model = None
        self.history = []
        self.theta = None
        self.score_notes = ''
        self.model_config = model_config

    def validate_trained(self):
        if not self._model:
            raise TypeError(
                self.log(f'The model has not yet been fit.  Precit() can only be called after the model has been fit.'))

    def vectorize_sequece(self, sequence, dimensions=10000):
        results = np.zeros((len(sequence), dimensions))
        for i, item in enumerate(sequence):
            results[i, item] = 1
        return results

    @property
    def coefficients(self):
        return self._model.coefficients