import numpy as np
import  pandas as pd
from sklearn.metrics import roc_curve, auc

from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.plotting import Plotter

class DataAnalysis_Quickstart(object):

    def run_basic_plots(self):
        data_config = DataConfig('sklearn', 'iris')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LogReg', 'sklearn', ModelType.LOGISTIC_REGRESSION, data_config)

        # 3. Create the Context.
        context_config = ContextConfig(model_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        # Use only two features for plotting
        # features = ('sepal length (cm)', 'sepal width (cm)')
        features = ('petal length (cm)', 'petal width (cm)')
        features = None
        context.prepare_data(features=features)


        # plot distributions
        df = pd.DataFrame(context.model.data_container.X, columns=context.model.data_container.X_columns)
        plotter = Plotter()
        plotter.plot_distributions(df, context.model.data_container.y)


        # plot correlations
        plotter.plot_heatmap(df.corr())
        plotter.show()

class Classifier_Quickstart(object):
    def __init__(self):
        pass

    def run_binary_logistic_regression(self, plot):
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
        context.prepare_data(features=features)

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
        # The ROC curve is for 1 class only, so we'll plot each class separately
        fpr, tpr, th = roc_curve(context.model.data_container.y_test, predictions.values)
        roc_auc = auc(fpr, tpr)
        plotter = Plotter()
        plotter.plot_roc_curve(fpr, tpr)
        plotter.show()


    def run_multiclass_logistic_regression(self, plot):
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
        context.prepare_data(features=features)

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

        # The contour plot will only be correct if 1 features is used!!!
        if plot:
            plotter = Plotter()
            plotter.plot_contour(context)
            plotter.show()


class Regression_Quickstart(object):
    def __init__(self):
        pass

    def run_linear_regression(self, plot=False, framework='sklearn'):
        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'diabetes')
        #data_config = DataConfig('sklearn', 'ca-housing')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LinReg', framework, ModelType.LINEAR_REGRESSION, data_config)

        # 3. Create the Context.
        context_config = ContextConfig(model_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        # To make this example clear, we'll use only 1 feature
        features = ('bmi')
        #features=('MedInc')
        #features = None
        # Set features to None or do not pass it as a parameter to prepare_data() if you'd like to train the model against all features.
        context.prepare_data(features=features)

        # 5. Train the model.
        context.train()
        print(f'Training fit the following coefficients: {context.model.coefficients}')

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.model.data_container.X_test, True)
        #print(f'Predictions - {predictions}')
        #print(context.model.data_container.y_test)

        # Plot outputs
        # The plot will only be correct if 1 features is used!!!
        if plot:
            import matplotlib.pyplot as plt
            plt.scatter(context.model.data_container.X_test,
                        context.model.data_container.y_test, c='k', marker='+')

            plt.plot(context.model.data_container.X_test, predictions.values, color='blue', linewidth=3)
            plt.xlabel(context.model.data_container.X_columns[0])
            plt.ylabel('Label')
            plt.title(f'{context.model.data_container.config.dataset_name} {context.model.model_config.name}')
            plt.show()

if __name__ == '__main__':
    quickstart = DataAnalysis_Quickstart()
    quickstart.run_basic_plots()

    #quickstart = Classifier_Quickstart()
    #quickstart.run_multiclass_logistic_regression(plot=True)

    #quickstart = Regression_Quickstart()
    #quickstart.run_linear_regression(plot=True, framework='tensorflow')