import numpy as np
import  pandas as pd
from sklearn.metrics import roc_curve, auc

from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig, VisualizationConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.visualization import Visualizer

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
        df = pd.DataFrame(context.data_container.X, columns=context.data_container.X_columns)
        vis = Visualizer()
        vis.plot_distributions(df, context.data_container.y)


        # plot correlations
        vis.plot_heatmap(df.corr())
        vis.show()

class Classifier_Quickstart(object):
    def __init__(self):
        pass

    def run_binary_logistic_regression(self, plot = False):
        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'iris')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LogReg', 'sklearn', ModelType.LOGISTIC_REGRESSION)

        # 3. Create the Context.
        context_config = ContextConfig(model_config, data_config)
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
        context.data_container.X = X
        context.data_container.y = y
        context.data_container.split()
        #context.prepare_data(features=features)

        # 5. Train the model.
        context.train()

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.data_container.X_test, with_probabilities=True)
        for name, prediction in predictions.items():
            # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.data_container.y_test]})
            results = zip(context.data_container.y_test, np.argmax(prediction.values, axis=1), prediction.values)
            for result in results:
                print(f'Label: {result[0]} Prediction: {result[1]} Model Output: {result[2]}')

        # Plot outputs
        if plot:
            vis = Visualizer()
            # The ROC curve is for 1 class only, so we'll plot each class separately
            for name, prediction in predictions.items():
                fpr, tpr, th = roc_curve(context.data_container.y_test, np.argmax(prediction.values, axis=1))
                roc_auc = auc(fpr, tpr)
                vis.plot_roc_curve(fpr, tpr, roc_auc, label=name)
            vis.show()


    def run_multiclass_logistic_regression(self, plot):
        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'iris')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LogReg', 'sklearn', ModelType.LOGISTIC_REGRESSION)

        # 3. Create the Context.
        context_config = ContextConfig(model_config, data_config)
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
        predictions = context.predict(context.data_container.X_test, with_probabilities=True)
        # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.data_container.y_test]})
        results = zip(context.data_container.y_test, np.argmax(predictions.values, axis=1), predictions.values)
        for result in results:
            print(f'Label: {result[0]} Prediction: {result[1]} Model Output: {result[2]}')

        # The contour plot will only be correct if 1 features is used!!!
        if plot:
            vis = Visualizer()
            vis.plot_contour(context)
            vis.show()


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
        print(f'Training fit the following coefficients: {context.coefficients}')

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.data_container.X_test, with_probabilities=True)
        #print(f'Predictions - {predictions}')
        #print(context.data_container.y_test)

        # Plot outputs
        # The plot will only be correct if 1 features is used!!!
        if plot:
            import matplotlib.pyplot as plt
            plt.scatter(context.data_container.X_test,
                        context.data_container.y_test, c='k', marker='+')

            plt.plot(context.data_container.X_test, predictions.values, color='blue', linewidth=3)
            plt.xlabel(context.data_container.X_columns[0])
            plt.ylabel('Label')
            plt.title(f'{context.data_container.config.dataset_name} {context.model.model_config.name}')
            plt.show()



class Multiple_Models_Quickstart(object):
    def __init__(self):
        pass

    def run_binary_classifier(self, plot=False):
        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'iris')

        # 2. Define the models.
        model_configs = []
        model_configs.append(ModelConfig('Sklearn LogReg', 'sklearn', ModelType.LOGISTIC_REGRESSION))
        model_configs.append(ModelConfig('Sklearn SVC', 'sklearn', ModelType.SVC))


        # 3. Create the Context.
        context_config = ContextConfig(model_configs, data_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        # Use only two features for plotting
        #features = ('sepal length (cm)', 'sepal width (cm)')
        features = ('petal length (cm)', 'petal width (cm)')

        '''
        plt.imshow(train_set_x_orig[index])

        ### START CODE HERE ### (≈ 2 lines of code)
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

        train_set_x = train_set_x_flatten / 255.
        test_set_x = test_set_x_flatten / 255.
        '''

        # override data management to turn multiclassification problem into binary classification
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris["data"][:, 3:]  # petal width
        y = (iris["target"] == 2).astype(np.int)
        context.data_container.X = X
        context.data_container.y = y
        context.data_container.split()
        #context.prepare_data(features=features)

        # 5. Train the model.
        context.train()

        #6 Evaluate the model.
        scores = context.evaluate()
        print('Scores:')
        for name, score in scores.items():
            print(f'\n\tScore[{name}] - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.data_container.X_test)
        for name, prediction in predictions.items():
            # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.data_container.y_test]})
            results = zip(context.data_container.y_test, prediction.values)
            for result in results:
                print(f'Label: {result[0]} Prediction: {result[1]}')

        # Plot outputs
        if plot:
            vis = Visualizer()
            # The ROC curve is for 1 class only, so we'll plot each class separately
            for name, prediction in predictions.items():
                fpr, tpr, th = roc_curve(context.data_container.y_test, prediction.values)
                roc_auc = auc(fpr, tpr)
                vis.plot_roc_curve(fpr, tpr, roc_auc, label=name)
            vis.show()

    def run_binary_classifier(self, plot=False):
        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'iris')

        # 2. Define the models.
        model_configs = []
        model_configs.append(ModelConfig('Sklearn LogReg', 'sklearn', ModelType.LOGISTIC_REGRESSION))
        model_configs.append(ModelConfig('Sklearn SVC', 'sklearn', ModelType.SVC))


        # 3. Create the Context.
        context_config = ContextConfig(model_configs, data_config)
        context_config.visualization_config = VisualizationConfig((8, 8), 'ggplot')
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        # Use only two features for plotting
        #features = ('sepal length (cm)', 'sepal width (cm)')
        features = ('petal length (cm)', 'petal width (cm)')

        '''
        plt.imshow(train_set_x_orig[index])

        ### START CODE HERE ### (≈ 2 lines of code)
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

        train_set_x = train_set_x_flatten / 255.
        test_set_x = test_set_x_flatten / 255.
        '''

        # override data management to turn multiclassification problem into binary classification
        from sklearn import datasets
        iris = datasets.load_iris()
        X = iris["data"][:, 3:]  # petal width
        y = (iris["target"] == 2).astype(np.int)
        context.data_container.X = X
        context.data_container.y = y
        context.data_container.split()
        #context.prepare_data(features=features)

        # 5. Train the model.
        context.train()

        #6 Evaluate the model.
        scores = context.evaluate()
        print('Scores:')
        for name, score in scores.items():
            print(f'\n\tScore[{name}] - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.data_container.X_test)
        for name, prediction in predictions.items():
            # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.data_container.y_test]})
            results = zip(context.data_container.y_test, prediction.values)
            for result in results:
                print(f'Label: {result[0]} Prediction: {result[1]}')

        # Plot outputs
        if plot:
            vis = Visualizer(context.config.visualization_config)
            # The ROC curve is for 1 class only, so we'll plot each class separately
            for name, prediction in predictions.items():
                fpr, tpr, th = roc_curve(context.data_container.y_test, prediction.values)
                roc_auc = auc(fpr, tpr)
                vis.plot_roc_curve(fpr, tpr, roc_auc, label=name)
            vis.show()


if __name__ == '__main__':
    #quickstart = DataAnalysis_Quickstart()
    #quickstart.run_basic_plots()

    #quickstart = Classifier_Quickstart()
    #quickstart.run_binary_logistic_regression(plot=True)
    #quickstart.run_multiclass_logistic_regression(plot=True)
    #quickstart.run_binary_image_classifier()

    #quickstart = Regression_Quickstart()
    #quickstart.run_linear_regression(plot=True, framework='tensorflow')

    quickstart = Multiple_Models_Quickstart()
    quickstart.run_binary_classifier(plot=True)