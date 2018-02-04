from rembrandtml.entities import MLEntityBase

class MLModelImplementation(MLEntityBase):
    def __init__(self):
        super(MLModelImplementation, self).__init__()
        self._reg = None
        self.theta = None

    def validate_trained(self):
        if not self._reg:
            raise TypeError(
                self.log(f'The model has not yet been fit.  Precit() can only be called after the model has been fit.'))
