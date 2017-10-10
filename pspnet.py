# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Reshape
from keras.layers import Permute
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.layers import merge, Convolution2D, UpSampling2D,Deconvolution2D,AtrousConvolution2D,ZeroPadding2D,Lambda,multiply,Conv2DTranspose

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

import keras.backend as K
from keras.utils import conv_utils
from keras.engine.topology import Layer
from keras.engine import InputSpec

class CroppingLike2D(Layer):
    def __init__(self, target_shape, offset=None, data_format=None,
                 **kwargs):
        """Crop to target.

        If only one `offset` is set, then all dimensions are offset by this amount.

        """
        super(CroppingLike2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.target_shape = target_shape
        if offset is None or offset == 'centered':
            self.offset = 'centered'
        elif isinstance(offset, int):
            self.offset = (offset, offset)
        elif hasattr(offset, '__len__'):
            if len(offset) != 2:
                raise ValueError('`offset` should have two elements. '
                                 'Found: ' + str(offset))
            self.offset = offset
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return (input_shape[0],
                    input_shape[1],
                    self.target_shape[2],
                    self.target_shape[3])
        else:
            return (input_shape[0],
                    self.target_shape[1],
                    self.target_shape[2],
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.int_shape(inputs)
        if self.data_format == 'channels_first':
            input_height = input_shape[2]
            input_width = input_shape[3]
            target_height = self.target_shape[2]
            target_width = self.target_shape[3]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))

            return inputs[:,
                          :,
                          self.offset[0]:self.offset[0] + target_height,
                          self.offset[1]:self.offset[1] + target_width]
        elif self.data_format == 'channels_last':
            input_height = input_shape[1]
            input_width = input_shape[2]
            target_height = self.target_shape[1]
            target_width = self.target_shape[2]
            if target_height > input_height or target_width > input_width:
                raise ValueError('The Tensor to be cropped need to be smaller'
                                 'or equal to the target Tensor.')

            if self.offset == 'centered':
                self.offset = [int((input_height - target_height) / 2),
                               int((input_width - target_width) / 2)]

            if self.offset[0] + target_height > input_height:
                raise ValueError('Height index out of range: '
                                 + str(self.offset[0] + target_height))
            if self.offset[1] + target_width > input_width:
                raise ValueError('Width index out of range:'
                                 + str(self.offset[1] + target_width))
            output = inputs[:,
                            self.offset[0]:self.offset[0] + target_height,
                            self.offset[1]:self.offset[1] + target_width,
                            :]
            return output

class BilinearUpSampling2D(Layer):
    """Upsampling2D with bilinear interpolation."""

    def __init__(self, target_shape=None,factor=None, data_format=None, **kwargs):
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {
            'channels_last', 'channels_first'}
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        self.target_shape = target_shape
        self.factor = factor
        if self.data_format == 'channels_first':
            self.target_size = (target_shape[2], target_shape[3])
        elif self.data_format == 'channels_last':
            self.target_size = (target_shape[1], target_shape[2])
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            return (input_shape[0], self.target_size[0],
                    self.target_size[1], input_shape[3])
        else:
            return (input_shape[0], input_shape[1],
                    self.target_size[0], self.target_size[1])

    def call(self, inputs):
        return K.resize_images(inputs, self.factor, self.factor, self.data_format)

    def get_config(self):
        config = {'target_shape': self.target_shape,
                'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block,dilation_rate=1,multigrid=[1,2,1],use_se=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'keras.., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if dilation_rate<2:
        multigrid = [1,1,1]

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a',dilation_rate=dilation_rate*multigrid[0])(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b',dilation_rate=dilation_rate*multigrid[1])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',dilation_rate=dilation_rate*multigrid[2])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    if use_se and stage<5:
        se = _squeeze_excite_block(x, filters3, k=1,name=conv_name_base+'_se')
        x = multiply([x, se])
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def _conv(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate',(1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    block = conv_params.setdefault("block", "assp")

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,activation='linear')(input)
        return conv
    return f


def aspp_block(x,num_filters=256,rate_scale=1,output_stride=16,input_shape=(512,512,3)):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv3_3_1 = ZeroPadding2D(padding=(6*rate_scale, 6*rate_scale))(x)
    conv3_3_1 = _conv(filters=num_filters, kernel_size=(3, 3),dilation_rate=(6*rate_scale, 6*rate_scale),padding='valid',block='assp_3_3_1_%s'%output_stride)(conv3_3_1)
    conv3_3_1 = BatchNormalization(axis=bn_axis,name='bn_3_3_1_%s'%output_stride)(conv3_3_1)
    
    conv3_3_2 = ZeroPadding2D(padding=(12*rate_scale, 12*rate_scale))(x)
    conv3_3_2 = _conv(filters=num_filters, kernel_size=(3, 3),dilation_rate=(12*rate_scale, 12*rate_scale),padding='valid',block='assp_3_3_2_%s'%output_stride)(conv3_3_2)
    conv3_3_2 = BatchNormalization(axis=bn_axis,name='bn_3_3_2_%s'%output_stride)(conv3_3_2)
    
    conv3_3_3 = ZeroPadding2D(padding=(18*rate_scale, 18*rate_scale))(x)
    conv3_3_3 = _conv(filters=num_filters, kernel_size=(3, 3),dilation_rate=(18*rate_scale, 18*rate_scale),padding='valid',block='assp_3_3_3_%s'%output_stride)(conv3_3_3)
    conv3_3_3 = BatchNormalization(axis=bn_axis,name='bn_3_3_3_%s'%output_stride)(conv3_3_3)
    

    # conv3_3_4 = ZeroPadding2D(padding=(24*rate_scale, 24*rate_scale))(x)
    # conv3_3_4 = _conv(filters=num_filters, kernel_size=(3, 3),dilation_rate=(24*rate_scale, 24*rate_scale),padding='valid')(conv3_3_4)
    # conv3_3_4 = BatchNormalization()(conv3_3_4)
    
    conv1_1 = _conv(filters=num_filters, kernel_size=(1, 1),padding='same',block='assp_1_1_%s'%output_stride)(x)
    conv1_1 = BatchNormalization(axis=bn_axis,name='bn_1_1_%s'%output_stride)(conv1_1)
    
    # global_feat = AveragePooling2D((input_shape[0]/output_stride,input_shape[1]/output_stride))(x)
    # global_feat = _conv(filters=num_filters, kernel_size=(1, 1),padding='same')(global_feat)
    # global_feat = BatchNormalization()(global_feat)
    # global_feat = BilinearUpSampling2D((256,input_shape[0]/output_stride,input_shape[1]/output_stride),factor=input_shape[1]/output_stride)(global_feat)
    
    y = merge([
        conv3_3_1,
        conv3_3_2,
        conv3_3_3,
        # conv3_3_4,
        conv1_1,
        # global_feat,
        ], mode='concat', concat_axis=3)
    
    # y = _conv_bn_relu(filters=1, kernel_size=(1, 1),padding='same')(y)
    y = _conv(filters=256, kernel_size=(1, 1),padding='same',block='assp_out_%s'%output_stride)(y)
    y = BatchNormalization(axis=bn_axis,name='bn_out_%s'%output_stride)(y)
    return y


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),dilation_rate=1,multigrid=[1,2,1],use_se=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    if dilation_rate>1:
        strides=(1,1)
    else:
        multigrid = [1,1,1]

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a',dilation_rate=dilation_rate*multigrid[0])(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b',dilation_rate=dilation_rate*multigrid[1])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c',dilation_rate=dilation_rate*multigrid[2])(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    if use_se and stage<5:
        se = _squeeze_excite_block(x, filters3, k=1,name=conv_name_base+'_se')
        x = multiply([x, se])
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x



def duc(x,factor=8,output_shape=(512,512,1)):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    H,W,c,r = output_shape[0],output_shape[1],output_shape[2],factor
    h = H/r
    w = W/r
    x = Conv2D(c*r*r, (3, 3),padding='same',name='conv_duc_%s'%factor)(x)
    x = BatchNormalization(axis=bn_axis,name='bn_duc_%s'%factor)(x)
    x = Activation('relu')(x)
    x = Permute((3,1,2))(x)
    x = Reshape((c,r,r,h,w))(x)
    x = Permute((1,4,2,5,3))(x)
    x = Reshape((c,H,W))(x)
    x = Permute((2,3,1))(x)
    
    return x


def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [new_height, new_width],
                                      align_corners=True)
    return resized


def interp_block(x,num_filters=512,level=1,input_shape=(512,512,3),output_stride=16):
    feature_map_shape = (input_shape[0]/output_stride,input_shape[1]/output_stride)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    if output_stride==16:
        scale = 5
    elif output_stride==8:
        scale = 10
    
    kernel = (level*scale,level*scale)
    strides = (level*scale,level*scale)
    global_feat = AveragePooling2D(kernel,strides=strides,name='pool_level_%s_%s'%(level,output_stride))(x)
    global_feat = _conv(filters=num_filters, kernel_size=(1, 1),padding='same',name='conv_level_%s_%s'%(level,output_stride))(global_feat)
    global_feat = BatchNormalization(axis=bn_axis,name='bn_level_%s_%s'%(level,output_stride))(global_feat)
    global_feat = Lambda(Interp, arguments={'shape': feature_map_shape})(global_feat)
    
    return global_feat 

def _squeeze_excite_block(input, filters, k=1,name=None):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    se_shape = (1, 1, filters * k) if K.image_data_format() == 'channels_last' else (filters * k, 1, 1)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense((filters * k) // 16, activation='relu', kernel_initializer='he_normal', use_bias=False,name=name+'_fc1')(se)
    se = Dense(filters * k, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,name=name+'_fc2')(se)
    return se



def pyramid_pooling_module(x,num_filters=512,input_shape=(512,512,3),output_stride=16,levels=[6,3,2,1]):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    pyramid_pooling_blocks = [x]
    for level in levels:
        pyramid_pooling_blocks.append(interp_block(x,num_filters=num_filters,level=level,input_shape=input_shape,output_stride=output_stride))

    # concat all these layers. resulted shape=(1,feature_map_size_x,feature_map_size_y,4096)
    y = merge(pyramid_pooling_blocks, mode='concat', concat_axis=3)
    y = _conv(filters=num_filters, kernel_size=(3, 3),padding='same',block='pyramid_out_%s'%output_stride)(y)
    y = BatchNormalization(axis=bn_axis,name='bn_pyramid_out_%s'%output_stride)(y)
    y = Activation('relu')(y)
    return y


def crop_deconv(classes, scale=1, kernel_size=(4, 4), strides=(2, 2),
               crop_offset='centered', weight_decay=0., block_name='featx'):
    """A VGG convolutional transpose block for decoding.

    :param classes: Integer, number of classes
    :param scale: Float, scale factor to the input feature, varing from 0 to 1
    :param kernel_size: Tuple, the kernel size for Conv2DTranspose layers
    :param strides: Tuple, the strides for Conv2DTranspose layers
    :param crop_offset: Tuple or "centered", the offset for cropping.
    The default is "centered", which crop the center of the feature map.

    >>> from keras_fcn.blocks import vgg_deconv
    >>> x = vgg_deconv(classes=21, scale=1e-2, block_name='feat2')(x)

    """
    def f(x, y):
        def scaling(xx, ss=1):
            return xx * ss
        scaled = Lambda(scaling, arguments={'ss': scale},
                        name='scale_{}'.format(block_name))(x)
        score = Conv2D(filters=classes, kernel_size=(1, 1),
                       activation='linear',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(weight_decay),
                       name='score_{}'.format(block_name))(scaled)
        if y is None:
            upscore = Conv2DTranspose(filters=classes, kernel_size=kernel_size,
                                      strides=strides, padding='valid',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay),
                                      use_bias=False,
                                      name='upscore_{}'.format(block_name))(score)
        else:
            crop = CroppingLike2D(target_shape=K.int_shape(y),
                                  offset=crop_offset,
                                  name='crop_{}'.format(block_name))(score)
            merge = add([y, crop])
            upscore = Conv2DTranspose(filters=classes, kernel_size=kernel_size,
                                      strides=strides, padding='valid',
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=l2(weight_decay),
                                      use_bias=False,
                                      name='upscore_{}'.format(block_name))(merge)
        return upscore
    return f

def PSPNet50(input_shape=(512,512,3),output_stride=16,num_blocks=4,multigrid=[1,1,1],levels=[6,3,2,1],use_se=True,upsample_type='bilinear'
             ):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = Input(shape=input_shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(
        64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),use_se=use_se)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',use_se=use_se)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',use_se=use_se)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',use_se=use_se)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',use_se=use_se)


    if output_stride==8:
        rate_scale=2
    elif output_stride==16:
        rate_scale=1


    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',dilation_rate=1*rate_scale,multigrid=multigrid,use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',dilation_rate=1*rate_scale,multigrid=multigrid,use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',dilation_rate=1*rate_scale,multigrid=multigrid,use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',dilation_rate=1*rate_scale,multigrid=multigrid,use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',dilation_rate=1*rate_scale,multigrid=multigrid,use_se=use_se)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',dilation_rate=1*rate_scale,multigrid=multigrid,use_se=use_se)

    init_rate = 2
    for block in range(4,num_blocks+1):
        if block==4:
            block=''
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a%s'%block,dilation_rate=init_rate*rate_scale,multigrid=multigrid,use_se=use_se)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b%s'%block,dilation_rate=init_rate*rate_scale,multigrid=multigrid,use_se=use_se)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c%s'%block,dilation_rate=init_rate*rate_scale,multigrid=multigrid,use_se=use_se)
        init_rate*=2
    
    # x1 = aspp_block(x,256,rate_scale=rate_scale,output_stride=output_stride,input_shape=input_shape)
    
    x = pyramid_pooling_module(x,num_filters=512,input_shape=input_shape,output_stride=output_stride,levels=levels)
    
    # x = merge([
    #         x1,
    #         x2,
    #         ], mode='concat', concat_axis=3)
    
    if upsample_type=='duc':
        x = duc(x,factor=output_stride,output_shape=(input_shape[0],input_shape[1],1))
        out = _conv(filters=1, kernel_size=(1, 1),padding='same',block='out_duc_%s'%output_stride)(x)
    elif upsample_type=='bilinear':
        x = _conv(filters=1, kernel_size=(1, 1),padding='same',block='out_bilinear_%s'%output_stride)(x)
        out = BilinearUpSampling2D((1,input_shape[0],input_shape[1]),factor=output_stride)(x)
    elif upsample_type=='deconv':
        out =  Conv2DTranspose(filters=1, kernel_size=(output_stride*2,output_stride*2),
                                          strides=(output_stride,output_stride), padding='same',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=None,
                                          use_bias=False,
                                          name='upscore_{}'.format('out'))(x)

    out = Activation('sigmoid')(out)

    model = Model(inputs=img_input, outputs=out)
    
    
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    model.load_weights(weights_path,by_name=True)
    print(model.summary())
    
    return model
