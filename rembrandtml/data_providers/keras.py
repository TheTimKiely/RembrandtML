from rembrandtml.data_providers.data_provider import DataProviderBase

def KerasDataProvider(DataProviderBase):
    def __init__(self, dataset_name):
        super(KerasDataProvider, self).__init__('keras', dataset_name)