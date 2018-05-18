import os
import sys
import argparse
import numpy as np
from keras import models

from rembrandtml.configuration import DataConfig, NeuralNetworkConfig, ContextConfig, RunMode
from rembrandtml.factories import ContextFactory
from rembrandtml.models import ModelType
from rembrandtml.utils import CommandLineParser, ImageProcessor


def parse_args(params):
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--architecture', help='Network architecture to use.')
    parser.add_argument('-b', '--batch_size', help='Training batch size')
    parser.add_argument('-c', '--test_data_source', help='Test data source, a directory will result in a generator')
    parser.add_argument('-d', '--data_source', help='Data source, a directory will result in a generator', required=True)
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to run')
    parser.add_argument('-f', '--framework', default='keras',
                        help='ML framework to use, e.g. ScikitLearn, TensorFlow, etc.')
    parser.add_argument('-i', '--image_size', help='Size of training images.')
    parser.add_argument('-m', '--model', help='Model file to use when testing or predicting')
    parser.add_argument('-o', '--output_dir', help='Output directory')
    parser.add_argument('-r', '--learning_rate', default=0.005, help='Gradient descent learning rate')
    parser.add_argument('-t', '--task', help='Task to perform, e.g. train, test, predict', default='train')
    return parser.parse_args(params)


def validate_required_params(task_name, given_params, required_params):
    error = ''
    for required_param in required_params:
        if not given_params[required_param]:
            error += 'The task \'{}\' requires the parameter \'{}\', which was not supplied.  '.format(task_name, required_param)
    if error:
        raise Exception(error)


def predict(args):
    validate_required_params('predict', vars(args), ('model',))
    model_path = args.model
    print(f'Loading model from: {model_path}')
    model = models.load_model(model_path)
    if model is None:
        raise TypeError(f'Failed to load model from file {model_path}')
    rgb_data = ImageProcessor(args.data_source).prepare_rgb_data(img_size=(128, 128))
    preds = model.predict(rgb_data)
    print(preds)


def main(params):
    # parse command line arguments
    args = parse_args(params)

    if args.task == 'predict':
        predict(args)
        return

    # 1. Define the datasource.
    # dataset = 'iris'
    # data_file = os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'gapminder', 'gm_2008_region.csv')))
    # data_config = DataConfig('pandas', dataset, data_file)
    #data_file = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'hab_train.h5'))
    #data_config = DataConfig('file', 'habs', data_source=args.data_file)
    data_config = DataConfig('generator', 'habs',
                             data_source='/home/timk/code/ML/projects/cyanotracker/images/raw/train/',
                             test_data_source='/home/timk/code/ML/projects/cyanotracker/images/raw/test/')

    # 2. Define the models.
    model_name = 'Keras Binary Classifier'
    weights_file_name = 'hab_' + model_name.replace(' ', '') + '_weights.h5'
    model_arch_file_name = 'hab_' + model_name.replace(' ', '') + '_model.json'
    model_file_name = 'hab_' + model_name.replace(' ', '') + '_model.h5'
    model_arch_file = os.path.abspath(os.path.join(args.output_dir, model_arch_file_name))
    model_file = os.path.abspath(os.path.join(args.output_dir, model_file_name))
    weights_file = os.path.abspath(os.path.join(args.output_dir, weights_file_name))
    model_configs = []
    model_configs.append(NeuralNetworkConfig(model_name, args.framework,
                                             ModelType.SIMPLE_CLASSIFICATION,
                                             (128, 128, 3),
                                             model_file, model_arch_file, weights_file,
                                             args.batch_size, args.epochs, args.learning_rate))

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
    predictions = context.predict(context.data_container.get_data(RunMode.EVALUATE))
    for prediction in predictions:
        print(predictions[prediction].values)
        print(predictions[prediction].accuracy)
    # for name, prediction in predictions.items():
    #    # df = pd.DataFrame({'Prediction': [[max(i) for i in predictions.values]], 'Predictions': [predictions.values], 'Labels:': [context.data_container.y_test]})
    #    results = zip(context.data_container.y_test, prediction.values)
    #    for result in results:
    #        print(f'Label: {result[0]} Prediction: {result[1]}')

    # Print train/test Errors
    # Predict test/train set examples
#    Y_pred_test = context.predict(context.data_container.get_data(RunMode.EVALUATE))
#    Y_pred_train = context.predict(context.data_container.get_data(RunMode.TRAIN))
#    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train[model_name].values -
#                                                             context.data_container.y_train)) * 100))
#    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test[model_name].values -
#                                                            context.data_container.y_test)) * 100))

if __name__ == '__main__':
    params = sys.argv[1:]
    main(params)

