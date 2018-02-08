import pandas as pd

class TestConfig(object):
    def __init__(self, model_name, log_file = None):
        self.model_name = model_name
        self.log_file = log_file

class RmlTest(object):
    def __init__(self):
        self.test_config = None
        self.results = pd.DataFrame({'Model': [], 'Score': []})

    def prepare_assert(self, expected, actual):
        """
        Creates a string that reports the expected and actual values that were passed to the assert
        :param expected:
        :param actual:
        :return:
        """
        return (expected, actual, f'Expected: {expected} Actual: {actual}')
