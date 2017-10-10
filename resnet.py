from __future__ import division

import six
import numpy as np
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Layer,
    Flatten,
    UpSampling2D,
    GlobalAveragePooling2D,
    Lambda,
    Reshape,
    Deconvolution2D,
    
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

import numpy as np
from keras import backend as K

# Create a 2D bilinear interpolation kernel in numpy for even size filters
def bilinear(w, h):
    import math
    data = np.zeros((w*h), dtype=float)
    f = math.ceil(w / 2.)
    c = (2 * f - 1 - f % 2) / (2. * f)
    # print ('f:{}, c:{}'.format(f, c))
    for i in range(w*h):
        x = float(i % w)
        y = float((i / w) % h)
        v = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        # print ('x:{}, y:{}, v:{}'.format(x, y, v))
        np.put(data, i, v)
    data = data.reshape((h, w))
    return data


# Create 4D bilinear interpolation kernel in numpy for even size filters
def bilinear4D(w, h, num_input_channels, num_filters):
    kern = bilinear(w, h)
    kern = kern.reshape((1, 1, w, h))
    kern = np.repeat(kern, num_input_channels, axis=0)
    kern = np.repeat(kern, num_filters, axis=1)
    for i in range(num_input_channels):
        for j in range(num_filters):
            if i != j:
                kern[i, j, :, :] = 0
    return kern


# Create a Keras bilinear weight initializer
def bilinear_init(shape, name=None, dim_ordering='th'):
    # print ('Shape: '),
    # print (shape)
    kernel = bilinear4D(shape[0], shape[1], shape[2], shape[3])
    np.set_printoptions(threshold=np.nan)
    kernel = kernel.transpose((2, 3, 0, 1))
    # print (kernel)
    # print (kernel.shape)
    kvar = K.variable(value=kernel, dtype=K.floatx(), name='bilinear')
    return kvar


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault('dilation_rate',(1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
    dilation_rate = conv_params.setdefault('dilation_rate',(1,1))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      dilation_rate=dilation_rate,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False,dilation_rate=1):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input,dilation_rate=dilation_rate):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
                if dilation_rate>1:
                    dilation_rate=1

            if dilation_rate!=1:
                init_strides = (1, 1)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0),dilation_rate=dilation_rate)(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False,dilation_rate=1):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           dilation_rate=dilation_rate,
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides,dilation_rate=dilation_rate)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3),dilation_rate=dilation_rate)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False,dilation_rate=1):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf

    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4),
                              dilation_rate=dilation_rate,
                              )(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     dilation_rate=dilation_rate,
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),dilation_rate=dilation_rate)(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1),dilation_rate=dilation_rate)(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


def aspp_block(x,num_filters=256):
    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Deconvolution2D,AtrousConvolution2D,ZeroPadding2D
    conv3_3_1 = ZeroPadding2D(padding=(6, 6))(x)
    conv3_3_1 = _conv_bn_relu(filters=num_filters, kernel_size=(3, 3),dilation_rate=(6, 6),padding='valid')(conv3_3_1)

    conv3_3_2 = ZeroPadding2D(padding=(12, 12))(x)
    conv3_3_2 = _conv_bn_relu(filters=num_filters, kernel_size=(3, 3),dilation_rate=(12, 12),padding='valid')(conv3_3_2)

    conv3_3_3 = ZeroPadding2D(padding=(18, 18))(x)
    conv3_3_3 = _conv_bn_relu(filters=num_filters, kernel_size=(3, 3),dilation_rate=(18, 18),padding='valid')(conv3_3_3)

    conv1_1 = _conv_bn_relu(filters=num_filters, kernel_size=(1, 1),padding='same')(x)

    # global_feat = AveragePooling2D((int(num_filters/2),int(num_filters/2)))(x)
    # # global_feat = Reshape((1,1,num_filters))(global_feat)
    # # global_feat = _conv_bn_relu(filters=num_filters, kernel_size=(1, 1),padding='same')(global_feat)
    # global_feat = UpSampling2D((num_filters,num_filters))(global_feat)
    # global_feat = _conv_bn_relu(filters=256, kernel_size=(1, 1),padding='same')(x)

    y = merge([conv3_3_1,conv3_3_2,conv3_3_3,conv1_1,x], mode='concat', concat_axis=3)
    y = _conv_bn_relu(filters=num_filters, kernel_size=(1, 1),padding='same')(y)
    return y



def bilinear_kernel(h,w,channels, use_bias = True, dtype = "float32") :
    """
    Returns uniform layer weights for a Conv2D or ConvTranspose2D layer in Keras 2.0, tensorflow channel ordering. 
    If the input image has size (in_height, in_width), the output image after scaling will have size
    (h*(in_height-1) + 1,  w*(in_width-1) + 1)
    Arguments:
        (h,w) : the magnification factors for height and weight. These must be positive integers.
        channels : the number of channels/filters, this must be the same in input and output image. Positive integer
        use_bias : must have the same value as 'use_bias' in the layer initialization. Default is True,
                  which means that the layer has a bias which is set to 0.0. Boolean
        dtype : the datatype of the returned weights
    """
    y = np.zeros((h,w,channels,channels), dtype = dtype)
    for i in range(0,h) :
        for j in range(0,w) :
            y[i,j,:,:] = np.identity(channels) / float(h*w*1)
    if use_bias : return [y,np.array([0.], dtype = dtype)]
    else : return [y]

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model


    @staticmethod
    def build_depplabv3(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)
        block0 = pool1
        #4 128
        block1_2 = _residual_block(block_fn, filters=64, repetitions=repetitions[0], is_first_layer=True)(block0)
        #8 64
        block2_3 = _residual_block(block_fn, filters=128, repetitions=repetitions[1], is_first_layer=False)(block1_2)
        #16 32
        block3_4 = _residual_block(block_fn, filters=256, repetitions=repetitions[2], is_first_layer=False)(block2_3)
        #16 32
        
        block4_5 = _residual_block(block_fn, filters=256, repetitions=repetitions[3], is_first_layer=True,dilation_rate=2)(block3_4)
        #16 32
        
        block_aspp = aspp_block(block4_5,256)     
        # out = Conv2D(256,1,1,activation='relu')(block_aspp)
        up0 = UpSampling2D((16,16))(block_aspp)
        out = Conv2D(filters=1, kernel_size=(1,1),
                  activation='sigmoid',padding='same')(up0)



        model = Model(inputs=input, outputs=out)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])


    @staticmethod
    def build_depplabv3_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build_depplabv3(input_shape, num_outputs, bottleneck, [3, 4, 6,3])

    @staticmethod
    def build_depplabv3_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build_depplabv3(input_shape, num_outputs, bottleneck, [3, 4, 23,3])



    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])