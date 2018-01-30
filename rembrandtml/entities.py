import os

from rembrandtml.configuration import Verbosity


class MLEntityBase(object):
    def __init__(self):
        self.Base_Directory = os.path.abspath(os.path.join(os.getcwd(), '../../..'))

    def unique_file_name(self, file_property, attribute_property):
        while(os.path.isfile(file_property.__get__(self))):
            attribute_property.__set__(self, attribute_property.__get__(self) + 1)
        return file_property.__get__(self)

    '''verbosity levels: s(silence) q(quiet) m(moderate) d(debug) (noisy)'''
    def log(self, msg, verbosity=Verbosity.DEBUG):
        '''
        Write the msg parameter to the console if the verbosity parameter is >= this objects configured verbosity in MLConfig.Verbosity
        :param msg:
        :param verbosity:
        :return: The msg parameter echoed back.  This allows nesting calls to log, e.g.raise TypeError(self.log(f'The features parameter was not supplied.')
        '''
        if(verbosity >= self.Config.Verbosity):
            print(msg)
        return msg

class MLContext(MLEntityBase):
    """The RMLContext object is an organizing structure to group the data, model, and plotter that are used in common ML tasks.
    It provides a collection of DataContainers, MLModels, and DataContainers to that that comparing entities becomes easy.
    """
    def __init__(self):
        super()
        self.plotters = {}
        self.data_containers = {}
        self.models = {}

    def plot(self, model_name = '', data_container_name = '', plotter_name = ''):
        pass




class Accuracy(object):
    def __init__(self):
        pass

class Prediction(object):
    def __init__(self):
        pass