import os

from rembrandtml.configuration import Verbosity
from rembrandtml.factories import ModelFactory
from rembrandtml.utils import MLLogger, Instrumentation


class MLEntityBase(object):
    '''
    Provided instrumentation functionality to all subclasses
    '''
    def __init__(self, instrumentation_config = None):
        self.Base_Directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.logger = MLLogger(instrumentation_config)

    def unique_file_name(self, file_property, attribute_property):
        while(os.path.isfile(file_property.__get__(self))):
            attribute_property.__set__(self, attribute_property.__get__(self) + 1)
        return file_property.__get__(self)

    # ToDo move to MLLogger
    def log(self, msg, verbosity=Verbosity.DEBUG):
        '''
        Write the msg parameter to the console if the verbosity parameter is >= this objects configured verbosity in MLConfig.Verbosity
        :param msg:
        :param verbosity:
        :return: The msg parameter echoed back.  This allows nesting calls to log, e.g.raise TypeError(self.log(f'The features parameter was not supplied.')
        '''
        self.logger.log(msg, verbosity)
        return msg

class MLContext(MLEntityBase):
    """The RMLContext object is an organizing structure to group the data, model, and plotter that are used in common ML tasks.
    It provides a collection of DataContainers, MLModels, and DataContainers to that that comparing entities becomes easy.
    """

    @staticmethod
    def create(config):
        '''
        Factory method to instantiate a new machine learning context
        :param MLCnfig:
        :return: MLContext
        '''
        model = ModelFactory.create('SkLearnKnn', config)
        context = MLContext(model, config)
        return context

    def __init__(self):
        super()
        self.plotters = {}
        self.data_containers = {}
        self.models = {}
        self.instrumentation = Instrumentation()

    def plot(self, model_name = '', data_container_name = '', plotter_name = ''):
        pass




class Accuracy(object):
    def __init__(self):
        pass

class Prediction(object):
    def __init__(self):
        pass