from rembrandtml.data_providers.data_provider import DataProviderBase

class SkLearnDataProvider(DataProviderBase):
    def __init__(self):
        super(SkLearnDataProvider, self).__init__('sklearn')