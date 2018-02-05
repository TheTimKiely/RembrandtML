from enum import Enum

class RunMode(Enum):
    TRAIN = 0
    EVALUATE = 1
    PREDICT = 2

class Verbosity(Enum):
    SILENT = 0
    QUIET = 1
    DEBUG = 2
    NOISY = 3

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented


class MLEntityBase(object):
    def __init__(self):
        self.Base_Directory = os.path.abspath(os.path.join(os.getcwd(), '../../..'))

    '''verbosity levels: s(silence) q(quiet) m(moderate) d(debug) (noisy)'''
    def log(self, msg, verbosity='d'):
        if(self.Config.Verbose == verbosity):
            print(msg)