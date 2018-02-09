from rembrandtml.core import MLContext
from rembrandtml.data import DataContainer
from rembrandtml.entities import MLLogger
from rembrandtml.model_implementations.model_impls_cntk import MLModelImplementationCntk
from rembrandtml.model_implementations.model_impls_keras import MLModelImplementationKeras
from rembrandtml.model_implementations.model_impls_sklearn import MLModelSkLearn
from rembrandtml.model_implementations.model_impls_tensorflow import MLModelTensorflow
from rembrandtml.models import MLModel, MathModel, ModelType
from rembrandtml.nnmodels import ConvolutionalNeuralNetwork, RecurrentNeuralNetwork, LstmRNN, GruNN
from rembrandtml.utils import Instrumentation


class ContextFactory(object):
    @staticmethod
    def create(config):
        '''
        Factory method to instantiate a new machine learning context
        :param MLCnfig:
        :return: MLContext
        '''

        logger = MLLogger(config.instrumentation_config)
        instrumentation = Instrumentation(config.instrumentation_config, logger)
        data_container = DataContainerFactory.create(config.data_config, instrumentation)
        model = ModelFactory.create('SkLearnKnn', config.model_config, data_container, instrumentation)
        context = MLContext(model, instrumentation, config)
        return context

class DataContainerFactory(object):
    @staticmethod
    def create(data_config, instrumentation):
        data_container = DataContainer(data_config, instrumentation)
        return data_container

class ModelImplFactory(object):
    @staticmethod
    def create(model_config, instrumentation):
        if  model_config.framework_name == 'sklearn':
            model_impl = MLModelSkLearn(model_config, instrumentation)
        elif model_config.framework_name == 'tensorflow':
            model_impl = MLModelTensorflow(model_config, instrumentation)
        elif model_config.framework_name == 'keras':
            model_impl = MLModelImplementationKeras(model_config, instrumentation)
        elif model_config.framework_name == 'cntk':
            model_impl = MLModelImplementationCntk(model_config, instrumentation)
        else:
            raise TypeError(f'The requested framework, {model_config.framework_name}, is not implemented.')
        return model_impl


class ModelFactory(object):
    @staticmethod
    def create(name, model_config, data_container, instrumentation):
        '''
        Factory method for creating ML models.
        This method first creates a DataContainer from the parameters specified in ContextConfig.DataConfig.
        :param name: The name of this model, e.g. SkLearnLinearRegression
        :param ml_config: An instance of ContextConfig, containing the parameters for this model and it's DataContainer.
        :return:
        '''
        # I'm not sure if a DataContain should be in __init__ for the models.
        # So, for now, we'll set the property
        if(model_config.model_type == ModelType.CNN):
            network = ConvolutionalNeuralNetwork(name,model_config, instrumentation)
        elif(model_config.model_type == ModelType.MATH):
            network = MathModel( name,model_config)
        elif model_config.model_type == ModelType.LINEAR_REGRESSION or \
                model_config.model_type == ModelType.SIMPLE_CLASSIFICATION or \
                model_config.model_type == ModelType.MULTIPLE_CLASSIFICATION or \
                model_config.model_type == ModelType.KNN or \
                model_config.model_type == ModelType.LOGISTIC_REGRESSION or \
                model_config.model_type == ModelType.SGD_CLASSIFIER or \
                model_config.model_type == ModelType.RANDOM_FOREST_CLASSIFIER or \
                model_config.model_type == ModelType.NAIVE_BAYES or \
                model_config.model_type == ModelType.PERCEPTRON or \
                model_config.model_type == ModelType.SVC or \
                model_config.model_type == ModelType.DECISTION_TREE_CLASSIFIER:
            model_impl = ModelImplFactory.create(model_config, instrumentation)
            network = MLModel(name, model_config, model_impl, instrumentation)
        elif model_config.model_type == ModelType.RNN:
            network = RecurrentNeuralNetwork( name,model_config, instrumentation)
        #elif(ml_config.model_type == 'DvsC'):
        #    network = ConvnetDogsVsCats( name,ml_config)
        elif model_config.model_type == ModelType.LSTM:
            network = LstmRNN( name,model_config, instrumentation)
        elif model_config.model_type ==  ModelType.GRU:
            network = GruNN( name,model_config, instrumentation)
        else:
            raise TypeError(f'Network type {model_config.model_type} is not defined.')
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



def multilabel_sample(y, size=1000, min_count=5, seed=None):
    """ Takes a matrix of binary labels `y` and returns
        the indices for a sample of size `size` if
        `size` > 1 or `size` * len(y) if size =< 1.
        The sample is guaranteed to have > `min_count` of
        each label.
    """
    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).all():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('multilabel_sample only works with binary indicator matrices')

    if (y.sum(axis=0) < min_count).any():
        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')

    if size <= 1:
        size = np.floor(y.shape[0] * size)

    if y.shape[1] * min_count > size:
        msg = "Size less than number of columns * min_count, returning {} items instead of {}."
        warn(msg.format(y.shape[1] * min_count, size))
        size = y.shape[1] * min_count

    rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))

    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(y.shape[0])

    sample_idxs = np.array([], dtype=choices.dtype)

    # first, guarantee > min_count of each label
    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])

    sample_idxs = np.unique(sample_idxs)

    # now that we have at least min_count of each, we can just random sample
    sample_count = int(size - sample_idxs.shape[0])

    # get sample_count indices from remaining choices
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices,
                                   size=sample_count,
                                   replace=False)

    return np.concatenate([sample_idxs, remaining_sampled])
