import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnLSTM
from tensorflow.contrib.rnn import LSTMBlockFusedCell


# deepcci neural network modules

def cnn_module(input, name, train, filters, kernel_size, pool_size, recomp=False):
    """
    CNN module consising of two convolutions, relu activations, residual skip connection, batch norm
    and pooling as described in paper
    :param input: input tensor for module
    :param name: name of module (used for variables and scope)
    :param train: is_train placeholder
    :param filters: number of convolutional filters
    :param kernel_size: size of convolutional filter kernels
    :param pool_size: pooling size
    :param recomp: whether used in recompute_gradient environment
    :return: output tensor
    """
    should_learn = tf.logical_and(train, tf.logical_not(recomp))
    with tf.variable_scope("pool_" + name):
        a = input
        a = tf.layers.conv1d(inputs=a, filters=filters, kernel_size=kernel_size, strides=1, padding='same',
                             name="conv1d_1",
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        a = tf.layers.batch_normalization(a, training=should_learn, name="bn_1")
        a = tf.nn.relu(a, name="relu")
        a = tf.layers.conv1d(inputs=a, filters=filters, kernel_size=kernel_size, strides=1, padding='same',
                             name="conv1d_2",
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
        a = tf.layers.batch_normalization(a, training=should_learn, name="bn_2")
        a = tf.nn.relu(a + input, name="relu_skip")
        a = tf.layers.max_pooling1d(inputs=a, pool_size=pool_size, strides=pool_size, padding="same", name="pool1d_1")
        return a


def cudnn_lstm_module(input, name, train, units, recomp=False):
    """
    CUDNN LSTM module
    :param input: input tensor
    :param name: name for variable / scope
    :param train: is_train placeholder
    :param units: number of LSTM units
    :param recomp: whether used in recompute_gradient environment
    :return: output tensor after LSTM
    """
    should_learn = tf.logical_and(train, tf.logical_not(recomp))

    class BiasInit:
        """
        Custom initialization for LSTM bias init
        """

        def __init__(self, init):
            self.count = 0
            self.init = init

        def __call__(self, shape, dtype):
            if self.count >= len(self.init):
                self.count = 0
            cop = tf.constant(self.init[self.count], dtype=dtype, shape=shape)
            self.count += 1
            return cop

    lstm = CudnnLSTM(
        num_layers=1,
        dtype=tf.float32,
        num_units=units,
        direction='unidirectional',
        name=name,
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
        bias_initializer=BiasInit([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # initialize forget gate bias to 1.0
        # according to [An empirical exploration of recurrent network architectures, Jozefowicz et al., ICML'15]
        # https://github.com/keras-team/keras/blob/04cbccc8038c105374eef6eb2ce96d6746999860/keras/layers/cudnn_recurrent.py#L448
    )
    a = input
    # lstm swaps batch and time dimension
    a = tf.transpose(a, perm=[1, 0, 2])
    a, c = lstm(a, training=True)
    # swap back lstm batch and time dimension
    a = tf.transpose(a, perm=[1, 0, 2])

    return a


def batch_norm(input, name, train, recomp=False):
    """
    batch normalization module wrapper
    :param input: input tensor
    :param name: name for variable scope
    :param train: is_train placeholder
    :param recomp: recompute_gradient environment
    :return:
    """
    should_learn = tf.logical_and(train, tf.logical_not(recomp))
    a = input
    a = tf.layers.batch_normalization(a, training=should_learn, name=name)
    return a


def fused_lstm_module(input, name, train, units, recomp=False):
    """
    tensorflow LSTM implementation - for inference only
    :param input: input tensor
    :param name: name for variable scope etc.
    :param train: is_train placeholder
    :param units: number of lstm units
    :param recomp: recompute_gradient environment
    :return: output tensor after LSTM
    """
    # only for inference due to forget_bias=0.0
    # CUDNN forget bias is trained without offset, but initialized to 1
    # fused forget bias trained with offset 1, but initialized to 0
    # fix: initialize fused forget bias to 1 and use forget_bias=0 to have compatibility with cudnn lstm

    should_learn = tf.logical_and(train, tf.logical_not(recomp))

    # manually set name to reflect cudnn weight names
    # forget bias = 0 needed as otherwise 1 added to already learned bias
    lstm = LSTMBlockFusedCell(num_units=units, forget_bias=0.0,
                              name=name + "/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell")
    # lstm swaps batch and time dimension
    input = tf.transpose(input, perm=[1, 0, 2])
    input, c = lstm(input, dtype=tf.float32)
    # swap back lstm batch and time dimension
    input = tf.transpose(input, perm=[1, 0, 2])

    return input


def resize(a, name, filters, kernel_size, strides, pool_size, pool_strides, relu, train, recomp=False):
    """
    resize layer comparable to resnet
    :param a: input tensor
    :param name: name for variable scope etc
    :param filters: number of convolutional filters (needed to remap feature count)
    :param kernel_size: filter size for convolution
    :param strides: convolutional stride
    :param pool_size: maxpooling size (if 1, no pooling used)
    :param pool_strides: maxpooling stride
    :param relu: whether relu activation should be added
    :param train: is_train placeholder
    :param recomp: recompute_gradient environment
    :return:
    """
    with tf.variable_scope(name):
        a = tf.layers.conv1d(inputs=a, filters=filters, kernel_size=kernel_size, strides=strides,
                             padding='same', name="conv1d_1")
        if relu:
            a = tf.nn.relu(a, name="relu")
        if not (pool_size == 1 and pool_strides == 1):
            a = tf.layers.max_pooling1d(inputs=a, pool_size=pool_size, strides=pool_strides, padding="same",
                                        name="pool1d_1")
    return a


def recompute(a, fn, name, train, func_params):
    """
    recompute gradient environment for saving memory - instead of saving intermediate results in memory, recompute when needed on backprop
    :param a: input tensor
    :param fn: actual module function
    :param name: name for scope
    :param train: is_train placeholder
    :param func_params: params for module function
    :return:
    """
    with tf.variable_scope('recompute_%s' % name, use_resource=True):
        a = tf.contrib.layers.recompute_grad(
            lambda x, is_recomputing: fn(x, recomp=is_recomputing, train=train, **func_params))(a)
    return a
