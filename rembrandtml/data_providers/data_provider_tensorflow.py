import numpy as np
import tensorflow as tf

from rembrandtml.data_providers.data_provider import DataProviderBase


class TensorflowDataProvider(DataProviderBase):
    def __init__(self, data_config, instrumentation):
        super(TensorflowDataProvider, self).__init__('tensorflow', data_config, instrumentation)

    def prepare_data(self, features=None, target_feature=None, sample_size=None):
        dataset = tf.contrib.learn.datasets.load_dataset(self.data_config.dataset_name)
        return None, dataset, None

    def split(self, X, y):
        # Tensorflow returns a data structure that is already divided into train and eval groups
        X_train = X.train.images
        y_train = np.asarray(X.train.labels, dbtype=np.int32)
        X_test = X.test.images
        y_test = np.asarray(X.test.labels, dbtype=np.int32)
