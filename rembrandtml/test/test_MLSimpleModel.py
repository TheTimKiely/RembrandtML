import os, sys

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.getcwd())
print(f'sys.path: {sys.path}')

from unittest import TestCase
import numpy as np
from sklearn.metrics import roc_curve, auc

from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig, NeuralNetworkConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.test.rml_testing import RmlTest
from rembrandtml.visualization import Visualizer


class TestMLSimpleModel(TestCase, RmlTest):

    def run_binary_classifier(self, framework_name, plot=False):
        # 1. Define the datasource.
        # dataset = 'iris'
        # data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        # data_config = DataConfig('pandas', dataset, data_file)
        data_file = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'hab_train.h5'))
        data_config = DataConfig('file', 'habs',
                                 data_file=data_file)

        # 2. Define the models.
        model_name = 'Keras Binary Classifier'
        weights_file_name = 'hab_' + model_name.replace(' ', '') + '_weights.h5'
        model_file_name = 'hab_' + model_name.replace(' ', '') + '_model.json'
        model_file = os.path.abspath(os.path.join(os.getcwd(), '..', 'models', model_file_name))
        weights_file = os.path.abspath(os.path.join(os.getcwd(), '..', 'models', weights_file_name))
        model_configs = []
        model_configs.append(NeuralNetworkConfig(model_name, framework_name,
                                                 ModelType.SIMPLE_CLASSIFICATION, model_file, weights_file,
                                                 10, 0.005))

        # 3. Create the Context.
        context_config = ContextConfig(model_configs, data_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        # Use only two features for plotting
        context.prepare_data()

        # 5. Train the model.
        context.train(save=True)

        # 6 Evaluate the model.
        #scores = context.evaluate()
        #print('Scores:')
        #for name, score in scores.items():
        #    print(f'\n\tScore[{name}] - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.data_container.X_test)
        #for name, prediction in predictions.items():
        #    # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.data_container.y_test]})
        #    results = zip(context.data_container.y_test, prediction.values)
        #    for result in results:
        #        print(f'Label: {result[0]} Prediction: {result[1]}')

        # Print train/test Errors
        # Predict test/train set examples
        Y_pred_test = context.predict(context.data_container.X_test)
        Y_pred_train = context.predict(context.data_container.X_train)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train[model_name].values -
                                                                 context.data_container.y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test[model_name].values -
                                                                context.data_container.y_test)) * 100))

        # Plot outputs
        if plot:
            vis = Visualizer()
            # The ROC curve is for 1 class only, so we'll plot each class separately
            for name, prediction in predictions.items():
                fpr, tpr, th = roc_curve(context.data_container.y_test, prediction.values)
                roc_auc = auc(fpr, tpr)
                vis.plot_roc_curve(fpr, tpr, roc_auc, label=name)
            vis.show()

    def test_binary_classifier_math(self, plot=False):
        self.run_binary_classifier('math')

    def test_binary_classifier_keras(self, plot=False):
        self.run_binary_classifier('keras')


if __name__ == '__main__':
    tests = TestMLSimpleModel()
    tests.test_binary_classifier_keras()