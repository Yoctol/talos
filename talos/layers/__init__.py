### Copy paste from
### https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/layers/__init__.py
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer

# Advanced activations.
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.advanced_activations import PReLU
from tensorflow.python.keras.layers.advanced_activations import ELU
from tensorflow.python.keras.layers.advanced_activations import ReLU
from tensorflow.python.keras.layers.advanced_activations import ThresholdedReLU
from tensorflow.python.keras.layers.advanced_activations import Softmax

# Convolution layers.
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.keras.layers.convolutional import Conv3DTranspose
from tensorflow.python.keras.layers.convolutional import SeparableConv1D
from tensorflow.python.keras.layers.convolutional import SeparableConv2D

# Convolution layer aliases.
from tensorflow.python.keras.layers.convolutional import Convolution1D
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.convolutional import Convolution3D
from tensorflow.python.keras.layers.convolutional import Convolution2DTranspose
from tensorflow.python.keras.layers.convolutional import Convolution3DTranspose
from tensorflow.python.keras.layers.convolutional import SeparableConvolution1D
from tensorflow.python.keras.layers.convolutional import SeparableConvolution2D
from tensorflow.python.keras.layers.convolutional import DepthwiseConv2D

# Image processing layers.
from tensorflow.python.keras.layers.convolutional import UpSampling1D
from tensorflow.python.keras.layers.convolutional import UpSampling2D
from tensorflow.python.keras.layers.convolutional import UpSampling3D
from tensorflow.python.keras.layers.convolutional import ZeroPadding1D
from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow.python.keras.layers.convolutional import ZeroPadding3D
from tensorflow.python.keras.layers.convolutional import Cropping1D
from tensorflow.python.keras.layers.convolutional import Cropping2D
from tensorflow.python.keras.layers.convolutional import Cropping3D

# Core layers.
from tensorflow.python.keras.layers.core import Masking
# overrided
# from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.core import SpatialDropout1D
from tensorflow.python.keras.layers.core import SpatialDropout2D
from tensorflow.python.keras.layers.core import SpatialDropout3D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.core import Reshape
from tensorflow.python.keras.layers.core import Permute
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import RepeatVector
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers.core import ActivityRegularization

# Locally-connected layers.
from tensorflow.python.keras.layers.local import LocallyConnected1D
from tensorflow.python.keras.layers.local import LocallyConnected2D

# Merge layers.
from tensorflow.python.keras.layers.merge import Add
from tensorflow.python.keras.layers.merge import Subtract
from tensorflow.python.keras.layers.merge import Multiply
from tensorflow.python.keras.layers.merge import Average
from tensorflow.python.keras.layers.merge import Maximum
from tensorflow.python.keras.layers.merge import Minimum
from tensorflow.python.keras.layers.merge import Concatenate
from tensorflow.python.keras.layers.merge import Dot
from tensorflow.python.keras.layers.merge import add
from tensorflow.python.keras.layers.merge import subtract
from tensorflow.python.keras.layers.merge import multiply
from tensorflow.python.keras.layers.merge import average
from tensorflow.python.keras.layers.merge import maximum
from tensorflow.python.keras.layers.merge import minimum
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers.merge import dot

# Noise layers.
from tensorflow.python.keras.layers.noise import AlphaDropout
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.layers.noise import GaussianDropout

# Normalization layers.
from tensorflow.python.keras.layers.normalization import BatchNormalization

# Pooling layers.
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D

from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling3D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling1D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling2D
from tensorflow.python.keras.layers.pooling import GlobalMaxPooling3D

# Pooling layer aliases.
from tensorflow.python.keras.layers.pooling import MaxPool1D
from tensorflow.python.keras.layers.pooling import MaxPool2D
from tensorflow.python.keras.layers.pooling import MaxPool3D
from tensorflow.python.keras.layers.pooling import AvgPool1D
from tensorflow.python.keras.layers.pooling import AvgPool2D
from tensorflow.python.keras.layers.pooling import AvgPool3D

from tensorflow.python.keras.layers.pooling import GlobalAvgPool1D
from tensorflow.python.keras.layers.pooling import GlobalAvgPool2D
from tensorflow.python.keras.layers.pooling import GlobalAvgPool3D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool1D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool2D
from tensorflow.python.keras.layers.pooling import GlobalMaxPool3D

# Recurrent layers.
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers.recurrent import StackedRNNCells
from tensorflow.python.keras.layers.recurrent import SimpleRNNCell
from tensorflow.python.keras.layers.recurrent import LSTMCell
from tensorflow.python.keras.layers.recurrent import SimpleRNN
from tensorflow.python.keras.layers.recurrent import GRU
from tensorflow.python.keras.layers.recurrent import LSTM

# Convolutional-recurrent layers.
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D

# Wrapper functions
from tensorflow.python.keras.layers.wrappers import Wrapper
from tensorflow.python.keras.layers.wrappers import Bidirectional
from tensorflow.python.keras.layers.wrappers import TimeDistributed

# Serialization functions
from tensorflow.python.keras.layers.serialization import deserialize
from tensorflow.python.keras.layers.serialization import serialize

### End copy paste

# Self-defined layers
from .conv1d_transpose import Conv1DTranspose
from .dropout import Dropout
from .embeddings import Embedding
from .layer_norm import LayerNormalization
from .positional_encode import PositionalEncode
from .recurrent import GRUCell

from .cudnn_recurrent import CuDNNLSTM
from .cudnn_recurrent import CuDNNGRU

import talos.layers.masking
import talos.layers.wrappers
