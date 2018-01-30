import os
from keras.applications import VGG16

class ModelRepository(object):

    @staticmethod
    def get_vgg16():
        #path = 'D:\code\ML\models\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        path = os.path.join(os.getcwd(), 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        path = 'imagenet'
        vgg16 = VGG16(weights=path, include_top=False, input_shape=(150, 150, 3))
        return vgg16