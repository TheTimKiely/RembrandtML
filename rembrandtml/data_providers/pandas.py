from rembrandtml.data_providers.data_provider import DataProviderBase

class PandasDataProvider(DataProviderBase):
    def __init__(self):
        super(PandasDataProvider, self).__init__('pandas')