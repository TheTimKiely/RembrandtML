import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType


class Classifier_Quickstart(object):
    def __init__(self):
        pass

    def make_meshgrid(self, x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(self, ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.values.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def plot_svm(self, context, predictions):
        import matplotlib.pyplot as plt
        X0 = context.model.data_container.X_test[:, 0]
        X1 = context.model.data_container.X_test[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)
        y = context.model.data_container.y_test
        fig = plt.figure()
        fig, ax = plt.subplots(1,1)
        self.plot_contours(ax, context, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title('Title')

        plt.show()
        '''
        # plt.scatter(np.argmax(predictions.values, axis=1), context.model.data_container.y_test)
        plt.scatter(context.model.data_container.X_test[:, 0], context.model.data_container.X_test[:, 1])
        plt.xlabel(context.model.data_container.X_columns[0])
        plt.ylabel(context.model.data_container.X_columns[1])
        w = context.model.coefficients[0]
        a = -w[0] / w[1]
        xx = np.linspace(-5, 5)
        yy = a * xx - (context.model.intercepts[0]) / w[1]

        # plt.plot(xx, yy, 'k-')

        plt.show()
        '''

    def run_logistic_regression(self, plot):
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
        context.prepare_data(features=('sepal length (cm)', 'sepal width (cm)'))

        # 5. Train the model.
        context.train()

        #6 Evaluate the model.
        score = context.evaluate()
        print(f'Score - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.model.data_container.X_test, True)
        print(f'Predictions - {predictions}')
        print(context.model.data_container.y_test)
        # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.model.data_container.y_test]})
        results = zip(context.model.data_container.y_test, np.argmax(predictions.values, axis=1), predictions.values)
        for result in results:
            print(f'Label: {result[0]} Prediction: {result[1]} Model Output: {result[2]}')

        # Plot outputs
        # The plot will only be correct if 1 features is used!!!
        if plot:
            self.plot_svm(context, predictions)


class Regression_Quickstart(object):
    def __init__(self):
        pass

    def run_linear_regression(self, plot=False):
        # 1. Define the datasource.
        #dataset = 'iris'
        #data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        #data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('sklearn', 'diabetes')

        # 2. Define the model.
        model_config = ModelConfig('Sklearn LinReg', 'sklearn', ModelType.LINEAR_REGRESSION, data_config)

        # 3. Create the Context.
        context_config = ContextConfig(model_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        # To make this example clear, we'll use only 1 feature
        features = ('bmi')
        # Set features to None or do not pass it as a parameter to prepare_data() if you'd like to train the model against all features.
        context.prepare_data(features)

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
            plt.xticks(())
            plt.yticks(())
            plt.show()
        #str = input('Press any key to continue...')


if __name__ == '__main__':
    #quickstart = Classifier_Quickstart()
    #quickstart.run_logistic_regression()

    quickstart = Regression_Quickstart()
    quickstart.run_linear_regression(plot=True)