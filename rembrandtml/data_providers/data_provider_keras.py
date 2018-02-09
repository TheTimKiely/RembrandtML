from rembrandtml.data_providers.data_provider import DataProviderBase

def KerasDataProvider(DataProviderBase):
    def __init__(self, data_config):
        super(KerasDataProvider, self).__init__('keras', data_config)