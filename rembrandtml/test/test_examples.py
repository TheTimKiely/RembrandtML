import logging, os
import numpy as np
from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig, InstrumentationConfig, \
    LoggingConfig, LoggerConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType


class RmlExamples(object):

    def logging_example(self):
        # 1. Define the datasource.
        data_config = DataConfig('sklearn', 'iris')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LogReg', 'sklearn', ModelType.LOGISTIC_REGRESSION, data_config)

        # 3. Create an InstrumentationConfig instance, giving it information about logging files and log levels
        # 3.a. Create a LoggerConfig
        # In this scenario, critical errors that require special attention are sent to a specific file
        logger_config = LoggerConfig(name='FileLogger',
                                     level=logging.DEBUG,
                                     handlers=(('debug.log', logging.DEBUG),
                                               ('Critical.log', logging.CRITICAL)),
                                     formatter='%(asctime)s %(levelname)-8s %(message)s')
        # 3.b. Create a LoggingConfig
        logging_config = LoggingConfig()
        # 3.c. Add the LoggerConfig to the LoggingConfig
        logging_config.logger_configs.append(logger_config)
        # 3.d. Creating IntrumentationConfig, passing the LoggingConfig as a parameter
        intrumentation_config = InstrumentationConfig(console_verbosity='d', logging_config=logging_config)

        # 4. Create the Context after adding the IntrumentationConfig to the ContextConfig
        context_config = ContextConfig(model_config)
        context_config.instrumentation_config = intrumentation_config
        context = ContextFactory.create(context_config)

        # 5. Prepare the data.
        # Use only two features for plotting
        # features = ('sepal length (cm)', 'sepal width (cm)')
        features = ('petal length (cm)', 'petal width (cm)')

        # override data management to turn multiclassification problem into binary classification
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris["data"][:, 3:]  # petal width
        y = (iris["target"] == 2).astype(np.int)
        context.model.data_container.X = X
        context.model.data_container.y = y
        context.model.data_container.split()
        # context.prepare_data(features=features)

        # 6. Train the model.
        context.train()

        # 7. Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 8. Make predictions.
        predictions = context.predict(context.model.data_container.X_test, True)
        # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.model.data_container.y_test]})
        results = zip(context.model.data_container.y_test, np.argmax(predictions.values, axis=1), predictions.values)
        for result in results:
            print(f'Label: {result[0]} Prediction: {result[1]} Model Output: {result[2]}')

    def tensorflow_cnn(self, plot = False):
        # 1. Define the datasource.
        data_config = DataConfig('tensorflow', 'mnist')

        # 2. Define the model.
        model_config = ModelConfig('Tensorflow CNN', 'tensorflow', ModelType.CNN, data_config)

        # 3. Create the Context.
        context_config = ContextConfig(model_config, base_dir=os.getcwd())
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        features = ('petal length (cm)', 'petal width (cm)')
        context.prepare_data()

        # 5. Train the model.
        context.train()

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.model.data_container.X_test, True)
        # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.model.data_container.y_test]})
        results = zip(context.model.data_container.y_test, np.argmax(predictions.values, axis=1), predictions.values)
        for result in results:
            print(f'Label: {result[0]} Prediction: {result[1]} Model Output: {result[2]}')


    def tensorflow_binary_classifier(self, plot = False):
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
        # Use only two features for plotting
        #features = ('sepal length (cm)', 'sepal width (cm)')
        features = ('petal length (cm)', 'petal width (cm)')

        # override data management to turn multiclassification problem into binary classification
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris["data"][:, 3:]  # petal width
        y = (iris["target"] == 2).astype(np.int)
        context.model.data_container.X = X
        context.model.data_container.y = y
        context.model.data_container.split()
        #context.prepare_data(features=features)

        # 5. Train the model.
        context.train()

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.model.data_container.X_test, True)
        # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.model.data_container.y_test]})
        results = zip(context.model.data_container.y_test, np.argmax(predictions.values, axis=1), predictions.values)
        for result in results:
            print(f'Label: {result[0]} Prediction: {result[1]} Model Output: {result[2]}')

        # Plot outputs
        if plot:
            # The ROC curve is for 1 class only, so we'll plot each class separately
            fpr, tpr, th = roc_curve(context.model.data_container.y_test, np.argmax(predictions.values, axis=1))
            roc_auc = auc(fpr, tpr)
            vis = Visualizer()
            vis.plot_roc_curve(fpr, tpr, roc_auc)
            vis.show()


if __name__ == '__main__':
    examples = RmlExamples()
    examples.tensorflow_cnn()