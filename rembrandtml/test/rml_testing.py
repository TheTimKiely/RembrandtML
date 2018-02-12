import os
import pandas as pd


class RmlTest(object):
    def __init__(self):
        self.run_config = None
        self.results = pd.DataFrame({'Model': [], 'Score': []})

    def get_data_file_path(self, data_dir, dataset_name, file_name):
        base_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))
        dir = os.path.abspath(os.path.join(base_directory, '..'))
        dir = os.path.join(dir, 'data', data_dir, dataset_name)
        return os.path.join(dir, file_name)

    def get_data_file_pathOLD(self, file_name):
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
