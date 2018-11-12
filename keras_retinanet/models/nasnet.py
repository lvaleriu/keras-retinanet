"""
Copyright 2018-2019 lvaleriu (https://github.com/lvaleriu/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import keras

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class NASNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return nasnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        if self.backbone == 'nasnet_large':
            weights_path = keras.utils.get_file(
                    'nasnet_large_no_top.h5',
                    keras.applications.nasnet.nasnet.NASNET_LARGE_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='d81d89dc07e6e56530c4e77faddd61b5')
        elif self.backbone == 'nasnet_mobile':
            weights_path = keras.utils.get_file(
                    'nasnet_mobile_no_top.h5',
                    keras.applications.nasnet.nasnet.NASNET_MOBILE_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='1ed92395b5b598bdda52abe5c0dbfd63')
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['nasnet_large', 'nasnet_mobile']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')


def nasnet_retinanet(num_classes, backbone='nasnet_large', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a NASNet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('nasnet_large', 'nasnet_mobile')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a NASNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the vgg backbone
    if backbone == 'nasnet_large':
        nasnet = keras.applications.NASNetLarge(input_tensor=inputs, include_top=False, pooling=None, weights=None)
    elif backbone == 'nasnet_mobile':
        nasnet = keras.applications.NASNetMobile(input_tensor=inputs, include_top=False, pooling=None, weights=None)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    bn_layers = [l for l in nasnet.layers if type(l) == keras.layers.normalization.BatchNormalization]
    for bn in bn_layers:
        bn.trainable = False
        
    if modifier:
        nasnet = modifier(nasnet)

    # create the full model   
    if backbone == 'nasnet_mobile':
        layer_names = ["normal_concat_3", "normal_concat_8", "normal_concat_12"]
    else:
        layer_names = ["normal_concat_5", "normal_concat_12", "normal_concat_18"]

    layer_outputs = [nasnet.get_layer(name).output for name in layer_names]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)
