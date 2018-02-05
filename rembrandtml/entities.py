import os

from rembrandtml.configuration import InstrumentationConfig, Verbosity
from rembrandtml.utils import Instrumentation


class MLEntityBase(object):
    """
    Provided instrumentation functionality to all subclasses
    """
    def __init__(self, instrumentation = None):
        self.Base_Directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
        if instrumentation == None:
            instrumentation = Instrumentation()
        self.instrumentation = instrumentation
        self.logger = MLLogger(instrumentation.config)

    def unique_file_name(self, file_property, attribute_property):
        while(os.path.isfile(file_property.__get__(self))):
            attribute_property.__set__(self, attribute_property.__get__(self) + 1)
        return file_property.__get__(self)

    # ToDo move to MLLogger
    def log(self, msg, verbosity=Verbosity.DEBUG):
        '''
        Write the msg parameter to the console if the verbosity parameter is >= this objects configured verbosity in ContextConfig.Verbosity
        :param msg:
        :param verbosity:
        :return: The msg parameter echoed back.  This allows nesting calls to log, e.g.raise TypeError(self.log(f'The features parameter was not supplied.')
        '''
        self.logger.log(msg, verbosity)
        return msg


class MLFile(MLEntityBase):
    def __init__(self):
        super(MLFile, self).__init__()
        self._base_dir = ''

    @property
    def BaseDir(self):
        return self._base_dir

    def unique_file_name(self):
        return 'file path'


class MLLogger(object):
    def __init__(self, instrumentation_config = None):
        if instrumentation_config:
            self.instrumentation_config = instrumentation_config
        else:
            self.instrumentation_config = InstrumentationConfig(Verbosity.DEBUG)

    def log(self, msg, verbosity = None):
        if (verbosity > self.instrumentation_config.verbosity):
            return
        instr_msg = f'{self.time.get_elapsed()} Split: {self.timer.get_split()}: {msg}'
        print(msg)