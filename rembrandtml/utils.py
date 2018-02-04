import time, getopt

from rembrandtml.core import MLEntityBase

from rembrandtml.configuration import ModelConfig, MLConfig, Verbosity, DataConfig, InstrumentationConfig


class MLFile(MLEntityBase):
    def __init__(self):
        super(MLFile, self).__init__()
        self._base_dir = ''

    @property
    def BaseDir(self):
        return self._base_dir

    def unique_file_name(self):
        return 'file path'

class CommandLineParser(object):
    @staticmethod
    def parse_command_line(params):
        layers = 3
        nodes = 16
        epochs = 10
        batch_size = 64
        nn_type = 'cnn'
        mode = 'p'
        verbosity = Verbosity.QUIET
        sample_size = 0
        dataset_name = ''
        framework = 'tensorflow'
        metrics = ['acc']
        model_config = ModelConfig()

        opts, args = getopt.getopt(params, shortopts='t:m:l:o:n:d:e:f:b:s:v:x:')
        for opt, arg in opts:
            if opt == '-b':
                batch_size = int(arg)
            elif opt == '-d':
                dataset_name = arg
            elif opt == '-e':
                epochs = int(arg)
            elif opt == '-f':
                framework = arg
            elif opt == '-l':
                model_config.loss_function = arg
            elif opt == '-o':
                model_config.optimizer = arg
            elif opt == '-m':
                mode = arg
            elif opt == '-n':
                nodes = int(arg)
            elif opt == '-s':
                sample_size = int(arg)
            elif opt == '-t':
                nn_type = arg
            elif opt == '-v':
                verbosity = Verbosity(arg)
            elif opt == 'x':
                metrics = arg

        ml_config = MLConfig(nn_type, framework, mode, layers, nodes, epochs, batch_size)
        model_config.metrics = metrics
        ml_config.model_config = model_config
        instr_config = InstrumentationConfig(verbosity)
        ml_config.instrumentation_config = instr_config
        data_config = DataConfig(dataset_name, sample_size)
        ml_config.data_config = data_config
        return ml_config

class MLLogger(object):
    def __init__(self, instrumentation_config = None):
        if instrumentation_config:
            self.instrumentation_config = instrumentation_config
        else:
            self.instrumentation_config = InstrumentationConfig(Verbosity.DEBUG)

    def log(self, msg, verbosity = None):
        if (verbosity >= self.instrumentation_config.verbosity):
            print(msg)

class Split(object):
    def __init__(self, name, start_time):
        self.Name = name
        self.StartTime = start_time

class Timer(object):
    def __init__(self):
        self.StartTime = None
        self.Splits = []

    def start(self):
        self.StartTime = time.time()

    def start_split(self, name):
        self.Splits.append(Split(name, time.time()))

    def get_split(self, name = None):
        if name == None:
            return time.time() - self.StartTime
        if self.Splits[name] == None:
            raise KeyError(f'The Split {name} has not been created.')
        return self.Splits[name].StartTime

class Instrumentation(object):
    def __init__(self):
        self.Timer = Timer()
