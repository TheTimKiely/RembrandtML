from rembrandtml.entities import MLEntityBase


class MLModelImplementation(MLEntityBase):
    def __init__(self, model_config, instrumentation):
        super(MLModelImplementation, self).__init__(instrumentation)
        self._model = None
        self.theta = None
        self.model_config = model_config

    def validate_trained(self):
        if not self._model:
            raise TypeError(
                self.log(f'The model has not yet been fit.  Precit() can only be called after the model has been fit.'))
