import pandas as pd

class TestConfig(object):
    def __init__(self, model_name, log_file = None):
        self.model_name = model_name
        self.log_file = log_file
        self.prediction_column = None
        self.prediction_index = None
        self.index_name = None

class RmlTest(object):
    def __init__(self):
        self.test_config = None
        self.results = pd.DataFrame({'Model': [], 'Score': []})


    def get_data_file_path(self, file_name):
        base_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
        dir = os.path.abspath(os.path.join(base_directory, '..'))
        dir = os.path.join(dir, 'kaggle', dataset_name)
        return os.path.join(dir, file_name)

    def prepare_assert(self, expected, actual):
        """
        Creates a string that reports the expected and actual values that were passed to the assert
        :param expected:
        :param actual:
        :return:
        """
        return (expected, actual, f'Expected: {expected} Actual: {actual}')
