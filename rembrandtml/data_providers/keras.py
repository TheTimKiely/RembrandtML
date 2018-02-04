from rembrandtml.data_providers.data_provider import DataProviderBase

def KerasDataProvider(DataProviderBase):
    def __init__(self):
        super(KerasDataProvider, self).__init__('keras')