import os
import sys
import argparse

from rembrandtml.configuration import DataConfig, NeuralNetworkConfig, ContextConfig
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.utils import CommandLineParser


def parse_args(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', help='Data file', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to run')
    parser.add_argument('-i', '--image_size', help='Size of training images.')
    parser.add_argument('-f', '--framework', default='keras',
                        help='ML framework to use, e.g. ScikitLearn, TensorFlow, etc.')
    parser.add_argument('-r', '--learning_rate', default=0.005, help='Gradient descent learning rate')
    parser.add_argument('-a', '--architecture', help='Network architecture to use.')
    return parser.parse_args(params)


def main(params):
    # parse command line arguments
    args = parse_args(params)

    # 1. Define the datasource.
    # dataset = 'iris'
    # data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
    # data_config = DataConfig('pandas', dataset, data_file)
    #data_file = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'hab_train.h5'))
    data_config = DataConfig('file', 'habs',
                             data_file=args.data_file)

    # 2. Define the models.
    model_name = 'Keras Binary Classifier'
    weights_file_name = 'hab_' + model_name.replace(' ', '') + '_weights.h5'
    model_arch_file_name = 'hab_' + model_name.replace(' ', '') + '_model.json'
    model_file_name = 'hab_' + model_name.replace(' ', '') + '_model.h5'
    model_arch_file = os.path.abspath(os.path.join(os.getcwd(), '..', 'models', model_arch_file_name))
    model_file = os.path.abspath(os.path.join(os.getcwd(), '..', 'models', model_file_name))
    weights_file = os.path.abspath(os.path.join(os.getcwd(), '..', 'models', weights_file_name))
    model_configs = []
    model_configs.append(NeuralNetworkConfig(model_name, args.framework,
                                             ModelType.SIMPLE_CLASSIFICATION, model_file, model_arch_file, weights_file,
                                             args.epochs, args.learning_rate))

    # 3. Create the Context.
    context_config = ContextConfig(model_configs, data_config)
    context = ContextFactory.create(context_config)

    # 4. Prepare the data.
    # Use only two features for plotting
    context.prepare_data()

    # 5. Train the model.
    context.train(save=True)

    # 6 Evaluate the model.
    # scores = context.evaluate()
    # print('Scores:')
    # for name, score in scores.items():
    #    print(f'\n\tScore[{name}] - {score}')

    # 7. Make predictions.
    predictions = context.predict(context.data_container.X_test)
    # for name, prediction in predictions.items():
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

if __name__ == '__main__':
    params = sys.argv[1:]
    main(params)