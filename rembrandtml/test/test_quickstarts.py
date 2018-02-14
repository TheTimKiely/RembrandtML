import os
from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType


class Classifier_Quickstart(object):
    def __init__(self):
        pass

    def run_logistic_regression(self):
        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'iris')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LogReg', 'sklearn', ModelType.LOGISTIC_REGRESSION, data_config)

        # 3. Create the Context.
        context_config = ContextConfig(model_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        context.prepare_data()

        # 5. Train the model.
        context.train()

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.model.data_container.X_test, True)
        print(f'Predictions - {predictions}')
        print(context.model.data_container.y_test)

class Regression_Quickstart(object):
    def __init__(self):
        pass

    def run_linear_regression(self):
        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'boston')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LinReg', 'sklearn', ModelType.LINEAR_REGRESSION, data_config)

        # 3. Create the Context.
        context_config = ContextConfig(model_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        context.prepare_data()

        # 5. Train the model.
        context.train()
        print(f'Training fit the following coefficients: {context.model.coefficients}')

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.model.data_container.X_test, True)
        print(f'Predictions - {predictions}')
        print(context.model.data_container.y_test)


if __name__ == '__main__':
    #quickstart = Classifier_Quickstart()
    #quickstart.run_logistic_regression()

    quickstart = Regression_Quickstart()
    quickstart.run_linear_regression()