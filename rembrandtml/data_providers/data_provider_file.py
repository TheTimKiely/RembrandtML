import os, h5py
import numpy as np

from rembrandtml.core import ParameterError
from rembrandtml.data_providers.data_provider import DataProviderBase

class FileDataProvider(DataProviderBase):
    def __init__(self, data_config, instrumentation):
        super(FileDataProvider, self).__init__('file', data_config, instrumentation)

    def validate_files(self):
        if self.data_config.data_file is None:
            raise ParameterError('A test file must be configured in the DataConfig')
        if not os.path.isfile(self.data_config.data_file):
            data_file = self.data_config.data_file
            print(f'The configured data file, {data_file}, was not found.')
            raise ParameterError(f'The configured data file, {data_file}, was not found.')

    def prepare_data(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        self.validate_files()
        dataset = h5py.File(self.data_config.data_file, "r")
        X_raw = np.array(dataset["data"][:])  # your train set features
        y = np.array(dataset["labels"][:])  # your train set labels

        #y = y_orig.reshape((1, y_orig.shape[0]))
        '''
        test_dataset = h5py.File(self.data_config.train_file, "r")
        X_test = np.array(test_dataset["data"][:])  # your test set features
        y_test = np.array(test_dataset["labels"][:])  # your test set labels
        '''
        classes = np.array(dataset["classes"][:])  # the list of classes
        columns = None

        #ToDO: How to property handle image data!!!!
        #X_flatten = X_raw.reshape(X_raw.shape[0], -1)
        #X = X_flatten / 255.

        return columns, X_raw, y