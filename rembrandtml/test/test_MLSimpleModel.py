from unittest import TestCase
import numpy as np
from sklearn.metrics import roc_curve, auc

from rembrandtml.configuration import DataConfig, ModelConfig, ContextConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.test.rml_testing import RmlTest
from rembrandtml.visualization import Visualizer


class TestMLSimpleModel(TestCase, RmlTest):
    def test_binary_classifier(self, plot=False):
        # 1. Define the datasource.
        # dataset = 'iris'
        # data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
        # data_config = DataConfig('pandas', dataset, data_file)
        data_config = DataConfig('file', 'habs',
                                 data_file='D:\code\ML\RembrandtML\data\hab_train.h5')

        # 2. Define the models.
        model_configs = []
        model_configs.append(ModelConfig('Simple Binary Classifier', 'simple', ModelType.SIMPLE_CLASSIFICATION))

        # 3. Create the Context.
        context_config = ContextConfig(model_configs, data_config)
        context = ContextFactory.create(context_config)

        # 4. Prepare the data.
        # Use only two features for plotting
        context.prepare_data()

        # 5. Train the model.
        context.train()

        # 6 Evaluate the model.
        scores = context.evaluate()
        print('Scores:')
        for name, score in scores.items():
            print(f'\n\tScore[{name}] - {score}')

        # 7. Make predictions.
        predictions = context.predict(context.data_container.X_test)
        #for name, prediction in predictions.items():
        #    # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.data_container.y_test]})
        #    results = zip(context.data_container.y_test, prediction.values)
        #    for result in results:
        #        print(f'Label: {result[0]} Prediction: {result[1]}')

        # Plot outputs
        if plot:
            vis = Visualizer()
            # The ROC curve is for 1 class only, so we'll plot each class separately
            for name, prediction in predictions.items():
                fpr, tpr, th = roc_curve(context.data_container.y_test, prediction.values)
                roc_auc = auc(fpr, tpr)
                vis.plot_roc_curve(fpr, tpr, roc_auc, label=name)
            vis.show()

if __name__ == '__main__':
    tests = TestMLSimpleModel()
    tests.test_binary_classifier()