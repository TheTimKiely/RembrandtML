import os, h5py
import numpy as np

from rembrandtml.core import ParameterError
from rembrandtml.data_providers.data_provider import DataProviderBase

class FileDataProvider(DataProviderBase):
    def __init__(self, data_config, instrumentation):
        super(FileDataProvider, self).__init__('file', data_config, instrumentation)

    def validate_files(self):
        if self.data_config.data_source is None:
            raise ParameterError('A test file must be configured in the DataConfig')
        if not os.path.isfile(self.data_config.data_source):
            data_file = self.data_config.data_source
            raise ParameterError(f'The configured data file, {data_file}, was not found.')

    def prepare_data(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        self.log(f'Using data file, {self.data_config.data_source}.')
        self.validate_files()
        dataset = h5py.File(self.data_config.data_source, "r")
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

class GeneratorDataProvider(DataProviderBase):
    def __init__(self, data_config, instrumentation):
        super(GeneratorDataProvider, self).__init__('generator', data_config, instrumentation)


    def build_generator(self, data, lookback, delay, min_index, max_index, shuffle, batch_size, step):
        if(max_index is None):
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randomint(min_index + lookback, max_index, size = batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)
            samples = np.zeros((len(rows),
                                lookback // step,
                                data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                targets[j]= data[rows[j] + delay][1]
            yield samples, targets

    def generator(self):
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(self.data_config.data_source,
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')
        return generator

    def test_generator(self):
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(rescale=1./255)
        generator = datagen.flow_from_directory(self.data_config.test_data_source,
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='binary')
        return generator

    def generatorOLD(self, data, lookback, delay, min_index, max_index, shuffle, batch_size, step):
        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros((len(rows),
                                lookback // step,
                                data.shape[-1]))
            targets = np.zeros((len(rows),))
            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data[indices]
                targets[j] = data[rows[j] + delay][1]
            #print(f'lookback:{lookback}, delay:{delay}, min_index{min_index}, max_index:{max_index}, shuffle{shuffle}, batch_size:{batch_size}, step: {step}')
            #print(f'Sample: {samples[0,0,0]} Target: {targets[0]}')
            yield samples, targets
