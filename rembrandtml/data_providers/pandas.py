from rembrandtml.data_providers.data_provider import DataProviderBase

class PandasDataProvider(DataProviderBase):
    def __init__(self, dataset_name):
        super(PandasDataProvider, self).__init__('pandas', dataset_name)


    def get_dataset(self, features=None, target_feature=None, sample_size=None):
        dataset = None
        return dataset