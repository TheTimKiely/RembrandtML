from keras import models
from keras import layers

from rembrandtml.data import DataContainer
from rembrandtml.models import MLModel, MathModel, ModelType
from rembrandtml.nnmodels import ConvolutionalNeuralNetwork, RecurrentNeuralNetwork, LstmRNN, GruNN


class ModelFactory(object):
    @staticmethod
    def create(name, ml_config):
        '''
        Factory method for creating ML models.
        This method first creates a DataContainer from the parameters specified in MLConfig.DataConfig.
        :param name: The name of this model, e.g. SkLearnLinearRegression
        :param ml_config: An instance of MLConfig, containing the parameters for this model and it's DataContainer.
        :return:
        '''
        data_container = DataContainer(ml_config.data_config.framework_name, ml_config.data_config.dataset_name)
        # I'm not sure if a DataContain should be in __init__ for the models.
        # So, for now, we'll set the property
        if(ml_config.model_type == ModelType.CNN):
            network = ConvolutionalNeuralNetwork(name,ml_config)
        elif(ml_config.model_type == ModelType.MATH):
            network = MathModel( name,ml_config)
        elif ml_config.model_type == ModelType.LINEAR_REGRESSION or \
                ml_config.model_type == ModelType.SIMPLE_CLASSIFICATION or \
                ml_config.model_type == ModelType.MULTIPLE_CLASSIFICATION:
            network = MLModel( name,ml_config)
        elif ml_config.model_type == ModelType.RNN:
            network = RecurrentNeuralNetwork( name,ml_config)
        #elif(ml_config.model_type == 'DvsC'):
        #    network = ConvnetDogsVsCats( name,ml_config)
        elif ml_config.model_type == ModelType.LSTM:
            network = LstmRNN( name,ml_config)
        elif ml_config.model_type ==  ModelType.GRU:
            network = GruNN( name,ml_config)
        else:
            raise TypeError(f'Network type {ml_config.model_type} is not defined.')
        network.data_container = data_container
        return network

class ModelBuilder(object):
    @staticmethod
    def build_model(cls, model_parameters):
        model = models.Sequential()
        for i in range(len(model_parameters.layers)):
            layer_params = model_parameters.layers[i]
            model.add(layers.Dense(layer_params.node_count,
                                   activation=layer_params.activation,
                                   input_shape=layer_params.input_shape))
        model.compile(optimizer=model_parameters.optimizer, loss=model_parameters.loss_function,
                      metrics=model_parameters.metrics)
        return model
