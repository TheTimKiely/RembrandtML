import os
from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType


class Classifier_Quickstart(object):
    def __init__(self):
        pass

    def run(self):
        # 1. Define the datasource.
        dataset = 'gapminder'
        data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        data_config = DataConfig('pandas', dataset, data_file)

        # 2. Define the model.
        model_config = ModelConfig(self.run_config.model_name, 'sklearn', ModelType.VOTING_CLASSIFIER, data_config)

        # 3. Create the Context.
        context_config = ContextConfig(model_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        context.prepare_data()

        # 5. Train the model.
        context.train()

        # 7. Make predictions.
        prediction = context.predict(context.model.data_container.X_test)


if __name__ == '__main__':
    quickstart = Classifier_Quickstart()
    quickstart.run()