from rembrandtml.models import *
from rembrandtml.nnmodels import *


class ModelFactory(object):
    @staticmethod
    def create(name, ml_config):
        if(ml_config.model_type == ModelType.CNN):
            network = ConvolutionalNeuralNetwork( name,ml_config)
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
