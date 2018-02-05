from rembrandtml.entities import MLEntityBase
from rembrandtml.plotting import Plotter
from rembrandtml.utils import Instrumentation


class Score(object):
    def __init__(self):
        pass


class Prediction(object):
    def __init__(self):
        pass


class MLContext(MLEntityBase):
    """The RMLContext object is an organizing structure to group the data, model, and plotter that are used in common ML tasks.
    It provides a collection of DataContainers, MLModels, and DataContainers to that that comparing entities becomes easy.
    """

    def __init__(self, model, data_provider, config):
        super()
        self.config = config
        self.plotter = Plotter()
        self.data_provider = data_provider
        self.model = model
        self.instrumentation = Instrumentation(config.instrumentation_config)

    def prepare_data(self):
        self.log(f'Preparing data with {str(self.data_provider)}')
        self.data_container.prepare_data()
        self.log(f'Finished preparing data with {str(self.data_provider)}')

    def train(self):
        self.log(f'Training model: {str(self.model)}')
        self.model.fit()
        self.log(f'Finished training model: {str(self.model)}')

    def evaluate(self):
        self.log(f'Training model: {str(self.model)}')
        score = self.model.evaluate()
        self.log(f'Finished training model: {str(self.model)}')
        return score

    def plot(self, model_name = '', data_container_name = '', plotter_name = ''):
        pass