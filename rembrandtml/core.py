from enum import Enum

from rembrandtml.configuration import Verbosity
from rembrandtml.entities import MLEntityBase
from rembrandtml.plotting import Plotter
from rembrandtml.utils import Instrumentation

class ScoreType(Enum):
    ACCURACY = 0
    R2 = 1
    PRECISION = 2
    RECALL = 3
    F1 = 4

class Score(object):
    def __init__(self, score_type, value):
        self.score_type = score_type
        self.value = value

    def __str__(self):
        return f'Score: {self.score_type.name} Value: {self.value}'


class Prediction(object):
    def __init__(self):
        pass


class MLContext(MLEntityBase):
    """The RMLContext object is an organizing structure to group the data, model, and plotter that are used in common ML tasks.
    It provides a collection of DataContainers, MLModels, and DataContainers to that that comparing entities becomes easy.
    """

    def __init__(self, model, config):
        super(MLContext, self).__init__(Instrumentation(config.instrumentation_config))
        self.config = config
        self.plotter = Plotter()
        self.model = model
        # Start the timer to that we can log elapsed times
        self.instrumentation.timer.start()

    def prepare_data(self):
        self.log(f'Preparing data with {str(self.model.data_container)}')
        self.model.data_container.prepare_data()
        self.log(f'Finished preparing data with {str(self.model.data_container)}')

    def train(self):
        """
        Trains the model against the training set in the model's DataContainer.  If split() has been called, DataContainer.X_train will be populated and that training set will be used.  Otherwise, the DataContainer.X will be used.
        :return:
        """
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

    def time_to_string(self, hrs, min, sec):
        return "%d:%02d:%02d" % (hrs, min, sec)

    def log(self, msg, verbosity = Verbosity.DEBUG):
        start_string = self.time_to_string(*self.instrumentation.timer.get_elapsed())
        split_string = self.time_to_string(*self.instrumentation.timer.get_split())
        instr_msg = f'{start_string} Split: {split_string}: {msg}'
        super(MLContext, self).log(instr_msg, verbosity)