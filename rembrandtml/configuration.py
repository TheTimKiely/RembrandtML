from rembrandtml.core import Verbosity


class DataConfig(object):
    def __init__(self, framework_name, dataset_name, sample_size = -1):
        self.framework_name = framework_name
        self.dataset_name = dataset_name
        self.sample_size = sample_size

class InstrumentationConfig(object):
    def __init__(self, verbosity):
        self.verbosity = verbosity

class MLConfig(object):
    def __init__(self, model_type, framework, mode, verbosity = Verbosity.QUIET, layers = 4, nodes = 16, epochs = 10, batch_size = 32):
        self._verbosity = verbosity
        self._model_type = model_type
        # Properties probably aren't necessary, so experimenting with public fields
        self.Layers = layers
        self.framework = framework
        self.Nodes = nodes
        self._epochs = epochs
        self.BatchSize = batch_size
        self.TrainDir = ''
        self.TestDir = ''
        self.ValidationDir = ''
        self.Mode = mode
        self.model_config = None
        self.data_config = None
        self.instrumentation_config = None

    @property
    def Verbosity(self):
        return self._verbosity

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, model_type):
        self._model_type = model_type

    @property
    def Epochs(self):
        return self._epochs


class ModelConfig(object):
    def __init__(self):
        self._metrics = []
        self._layers = []
        self.LayerCount = 1
        self.Dropout = 0
        self.RecurrentDropout = 0


    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function):
        self._loss_function = loss_function

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        self._epochs = epochs

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def validation_data(self):
        return self._validation_data
    @validation_data.setter
    def validation_data(self, validation_data):
        self._validation_data = validation_data

    @property
    def metrics(self):
        return self._metrics
    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def layers(self):
        return self._layers
    @layers.setter
    def layers(self, layers):
        self._layers = layers
