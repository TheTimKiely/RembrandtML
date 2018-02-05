import time, getopt

from rembrandtml.configuration import ModelConfig, ContextConfig, DataConfig, InstrumentationConfig

class CommandLineParser(object):
    @staticmethod
    def parse_command_line(params):
        layers = 3
        nodes = 16
        epochs = 10
        batch_size = 64
        nn_type = 'cnn'
        mode = 'p'
        verbosity_code = 'q'
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
                verbosity_code = arg
            elif opt == 'x':
                metrics = arg

        ml_config = ContextConfig(nn_type, framework, mode, layers, nodes, epochs, batch_size)
        model_config.metrics = metrics
        ml_config.model_config = model_config
        instr_config = InstrumentationConfig(verbosity_code)
        ml_config.instrumentation_config = instr_config
        data_config = DataConfig(dataset_name, sample_size)
        ml_config.data_config = data_config
        return ml_config


class Split(object):
    def __init__(self, name, start_time):
        self.Name = name
        self.StartTime = start_time

class Timer(object):
    def __init__(self):
        self.start_time = None
        self.splits = []

    def get_start(self):
        self.start_time = time.time()

    def get_elapsed(self):
        return time.time() - self.get_start()

    def start_split(self, name):
        self.splits.append(Split(name, time.time()))

    def get_split(self, name = None):
        if name == None:
            return time.time() - self.start_time
        if self.splits[name] == None:
            raise KeyError(f'The Split {name} has not been created.')
        return self.splits[name].StartTime

class Instrumentation(object):
    def __init__(self, instrumentation_config = None):
        self.Timer = Timer()
        if instrumentation_config == None:
            config = InstrumentationConfig()
        self.config = instrumentation_config
