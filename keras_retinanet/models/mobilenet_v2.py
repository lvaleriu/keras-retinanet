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
from keras.applications import mobilenet_v2
from keras.utils import get_file
from ..utils.image import preprocess_image

from . import retinanet
from . import Backbone


class MobileNetV2Backbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    allowed_backbones = [96, 128, 160, 192, 224]

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return mobilenet_v2_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Download pre-trained weights for the specified backbone name.
        This name is in the format mobilenet{rows}_{alpha} where rows is the
        imagenet shape dimension and 'alpha' controls the width of the network.
        For more info check the explanation from the keras mobilenet script itself.
        """

        alpha = float(self.backbone.split('_')[2])
        rows  = int(self.backbone.split('_')[3].replace('mobilenet_v2', ''))

        # load weights
        if keras.backend.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
            
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
                raise ValueError('Alpha can be one of `0.35`, `0.50`, `0.75`, '
                             '`1.0`, `1.3` or `1.4` only.')

        model_name = 'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{}_{}_no_top.h5'.format(alpha, rows)
        weights_url = mobilenet_v2.mobilenet_v2.BASE_WEIGHT_PATH + model_name
        weights_path = get_file(model_name, weights_url, cache_subdir='models')

        return weights_path

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        backbone = int(self.backbone.split('_')[3])

        if backbone not in MobileNetV2Backbone.allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, MobileNetBackboneV2.allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
            We assume inputs to be BGR already.
            This function applies the "Inception" preprocessing which converts
            the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
            function is different from `imagenet_utils.preprocess_input().
        """

        x = inputs.astype(keras.backend.floatx())

        # 'BGR'->'RGB'
        x = x[..., ::-1]
        x /= 128
        x -= 1.

        return x


def mobilenet_v2_retinanet(num_classes, backbone='mobilenet_v2_1.0_224', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a mobilenet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('mobilenet_v2_96', mobilenet_v2_128', 'mobilenet_v2_160', 'mobilenet_v2_192', 'mobilenet_v2_224')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a MobileNet backbone.
    """
    alpha = float(backbone.split('_')[2])

    # choose default input
    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    backbone = mobilenet_v2.MobileNetV2(input_tensor=inputs, alpha=alpha, include_top=False, pooling=None, weights=None)
    bn_layers = [l for l in backbone.layers if type(l) == keras.layers.normalization.BatchNormalization]
    for bn in bn_layers:
        bn.trainable = False
        
    # create the full model
    layer_names   = ['block_4_project', 'block_10_project', 'block_16_project']
    layer_outputs = [backbone.get_layer(name).output for name in layer_names]
    backbone = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)

    # invoke modifier if given
    if modifier:
        backbone = modifier(backbone)

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone.outputs, **kwargs)
