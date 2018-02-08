from enum import Enum

from rembrandtml.configuration import Verbosity
from rembrandtml.entities import MLEntityBase
from rembrandtml.plotting import Plotter
from rembrandtml.utils import Instrumentation

class ScoreType(Enum):
    LOSS = 0
    ACCURACY = 1
    R2 = 2
    PRECISION = 3
    RECALL = 4
    F1 = 5

class Score(object):
    def __init__(self, model_config, score_type, value):
        self.score_type = score_type
        self.value = value
        self._model_config = model_config
        self.metrics = {}

    def __gt__(self, other):
        return self.value > other.value


    def __lt__(self, other):
        return self.value < other.value

    @property
    def model_name(self):
        return self._model_config.name


    @property
    def model_type(self):
        return self._model_config.model_type


    @property
    def model_framework(self):
        return self._model_config.framework_name

    def __str__(self):
        return f'Name:{self.model_name} Model Type: {self.model_type} Framework:{self.model_framework}\n\t' \
               f'Score: {self.score_type.name} Value: {self.value}\n\t' \
               f'Metrics: {self.metrics}'


class TuningResults(object):
    def __init__(self, model_name, best_params):
        self.model_name = model_name
        self.best_params = best_params


class Prediction(object):
    def __init__(self):
        pass


class MLContext(object):
    """The RMLContext object is an organizing structure to group the data, model, and plotter that are used in common ML tasks.
    It provides a collection of DataContainers, MLModels, and DataContainers to that that comparing entities becomes easy.
    The MLContext also holds singletons, such as Instrumentation and Timer.
    """

    def __init__(self, model, instrumentation, config):
        super(MLContext, self).__init__()
        self.config = config
        self.plotter = Plotter()
        self.model = model
        # Start the timer to that we can log elapsed times
        self.instrumentation = instrumentation
        self.instrumentation.timer.start()

    def prepare_data(self, features=None, target_feature=None):
        self.log(f'Preparing data with {str(self.model.data_container)}')
        self.model.data_container.prepare_data(features, target_feature)
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
        self.log(f'Evaluating model: {str(self.model)}')
        score = self.model.evaluate()
        self.log(f'Finished evaluating model: {str(self.model)}')
        return score

    def tune(self, tuning_parameters, model_parameters):
        """
        Tunes the model's hyperparameters.
        :param parameters: A dictionary of hyperparameters to be used.
        :return: A TuningResults instance.
        """
        self.log(f'Tuning model: {str(self.model)}')
        results = self.model.tune(tuning_parameters, model_parameters)
        self.log(f'Finished tuning model: {str(self.model)}')
        return results


    def predict(self, X):
        self.log(f'Predicting: {str(self.model)}')
        prediction = self.model.predict(X)
        self.log(f'Finished predicting: {str(self.model)}')
        return prediction

    def plot(self, model_name = '', data_container_name = '', plotter_name = ''):
        pass

    def time_to_string(self, hrs, min, sec):
        return "%d:%02d:%02d" % (hrs, min, sec)

    def log(self, msg, verbosity = Verbosity.DEBUG):
        start_string = self.time_to_string(*self.instrumentation.timer.get_elapsed())
        split_string = self.time_to_string(*self.instrumentation.timer.get_split())
        instr_msg = f'{start_string} Split: {split_string}: {msg}'
        self.instrumentation.logger.log(instr_msg, verbosity)

class FunctionNotImplementedError(NotImplementedError):
    def __init__(self,  type_name, function_name):
        super(FunctionNotImplementedError, self).__init__(self, f'{type_name}.{function_name}')