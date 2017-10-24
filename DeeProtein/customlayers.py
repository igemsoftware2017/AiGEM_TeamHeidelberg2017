import tensorflow as tf
import tensorlayer as tl
import math
import helpers


def focal_lossIII(prediction_tensor,
                  target_tensor,
                  weights,
                  gamma=2.,
                  epsilon=0.00001,
                  ):
    """Compute loss function.

    This function was adapted from the Tensorflow issues section on GitHub.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape [batch_size, num_anchors]
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a (scalar) tensor representing the value of the loss function
            or a float tensor of shape [batch_size, num_anchors]
    """
    preds = tf.nn.sigmoid(prediction_tensor)
    preds = tf.where(tf.equal(target_tensor, 1), preds, 1. - preds)
    losses = (1. - preds) ** gamma * tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor,
                                                                             logits=prediction_tensor)
    return losses


def resnet_block(inlayer, channels=[128, 256], pool_dim=2, summary_collection=None, is_train=True, name='scope'):
    """Define a residual block for DeeProtein.

    A residual block consists of two 1d covolutional layers both with a kernel size of 3 and a 1x1 1d convolution.
    Every conv layer is followed by a BatchNorm layer. The input may be pooled (optional).

    Args:
      inlayer: A `tl.layer` object holding the input.
      channels: A `Array` defining the channels.
      pool_dim:  A `int32` defining the pool dims, defaults to 2. May be None (no pooling).
      is_train: A `bool` from which dataset (train/valid) to draw the samples.
      summary_collection: A `str` object defining the collection to which to attach the summaries of the layers.
        Defaults to `None`.
      name: A `str` defining the scope to attach to te resBlock. Scopes must be unique in the network.

    Returns:
      A `tl.layer` object holding the Residual Block.
    """
    # calculate the block
    with tf.variable_scope(name) as vs:
        with tf.variable_scope('conv1') as vs:
            conv = tl.layers.Conv1dLayer(inlayer,
                                         act=prelu,
                                         shape=[3, channels[0], channels[1]],  # 32 features for each 5x5 patch
                                         stride=1,
                                         padding='SAME',
                                         W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                         W_init_args={},
                                         b_init=tf.constant_initializer(value=0.1),
                                         b_init_args={},
                                         name='cnn_layer')
            helpers._add_var_summary(conv.all_params[-2], 'conv', collection=summary_collection)

            norm = tl.layers.BatchNormLayer(conv, decay=0.9, epsilon=1e-05,
                                            is_train=is_train,
                                            name='batchnorm_layer')
        with tf.variable_scope('conv2') as vs:
            conv = tl.layers.Conv1dLayer(norm,
                                         act=prelu,
                                         shape=[3, channels[1], channels[1]*2],  # 32 features for each 5x5 patch
                                         stride=1,
                                         padding='SAME',
                                         W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                         W_init_args={},
                                         b_init=tf.constant_initializer(value=0.1),
                                         b_init_args={},
                                         name='cnn_layer')
            norm = tl.layers.BatchNormLayer(conv, decay=0.9, epsilon=1e-05,
                                            is_train=is_train,
                                            name='batchnorm_layer')
        with tf.variable_scope('1x1') as vs:
            conv = tl.layers.Conv1dLayer(norm,
                                         act=prelu,
                                         shape=[1, channels[1]*2, channels[1]],  # 32 features for each 5x5 patch
                                         stride=1,
                                         padding='SAME',
                                         W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                         W_init_args={},
                                         b_init=tf.constant_initializer(value=0.1),
                                         b_init_args={},
                                         name='1x1_layer')
            y = tl.layers.BatchNormLayer(conv, decay=0.9, epsilon=1e-05,
                                         is_train=is_train,
                                         name='batchnorm_layer')
        if pool_dim:
            with tf.variable_scope('pool') as vs:
                y = PoolLayer1d(y, ksize=[pool_dim], strides=[pool_dim], padding='VALID', pooling_type="MAX", name='pool_layer')

        with tf.variable_scope('shortcut') as vs:
            # reduce the shortcut
            if pool_dim:
                shortcut = PoolLayer1d(inlayer, ksize=[pool_dim], strides=[pool_dim], padding='VALID', pooling_type="MAX", name='pool_layer')
            else:
                shortcut = inlayer
            # zero pad the channels
            if channels[0] != channels[1]:
                paddings = [[0,0],
                            [0,0],
                            [0, channels[1]-channels[0]]
                            ]
                shortcut = PadLayer(shortcut, paddings=paddings)

            out = tl.layers.ElementwiseLayer([y, shortcut],
                                             combine_fn=tf.add,
                                             name='merge')
            return out

def prelu(x):
    """Calculate a parameterized rectified linear unit.

    Simple pRELU implementation with a weigth of 0.2.

    Args:
      x: A `Tensor` of which to calcualte the pRELU.

    Returns:
      A `Tensor` of same shape and type as input.
    """
    v = tf.get_variable("prelu_weight", dtype=tf.float32, initializer=tf.constant(0.2))
    return tf.where(tf.greater_equal(x, 0), x, x*v)


def list_remove_repeat(l=None):
    """A COPY FROM THE TENSORLAYER SRC. Remove the repeated items in a list, and return the processed list.
    You may need it to create merged layer like Concat, Elementwise and etc.

    Args:
      l: a list

    Returns:
      A non redundant version of that list.
    """
    l2 = []
    [l2.append(i) for i in l if not i in l2]
    return l2


Layer = tl.layers.Layer


class PadLayer(Layer):
    """A COPY FROM THE TENSORLAYER SRC.
    The :class:`PadLayer` class is a Padding layer for any modes and dimensions.
    Please see `tf.pad <https://www.tensorflow.org/api_docs/python/tf/pad>`_ for usage.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    padding : a Tensor of type int32.
    mode : one of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive)
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
            self,
            layer = None,
            paddings = None,
            mode = 'CONSTANT',
            name = 'pad_layer',
    ):
        Layer.__init__(self, name=name)
        assert paddings is not None, "paddings should be a Tensor of type int32. see https://www.tensorflow.org/api_docs/python/tf/pad"
        self.inputs = layer.outputs
        print("  [TL] PadLayer   %s: paddings:%s mode:%s" %
              (self.name, list(paddings), mode))

        self.outputs = tf.pad(self.inputs, paddings=paddings, mode=mode, name=name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )


def flatten_unknown(variable, name=''):
    """A COPY FROM THE TENSORLAYER SRC. Reshapes high-dimension input to a vector.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    variable : a tensorflow variable
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
     W_conv2 = weight_variable([5, 5, 100, 32])   # 64 features for each 5x5 patch
     b_conv2 = bias_variable([32])
     W_fc1 = weight_variable([7 * 7 * 32, 256])

     h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
     h_pool2 = max_pool_2x2(h_conv2)
     h_pool2.get_shape()[:].as_list() = [batch_size, 7, 7, 32]
             [batch_size, mask_row, mask_col, n_mask]
     h_pool2_flat = tl.layers.flatten_reshape(h_pool2)
             [batch_size, mask_row * mask_col * n_mask]
     h_pool2_flat_drop = tf.nn.dropout(h_pool2_flat, keep_prob)

    """
    dim = tf.reduce_prod(tf.shape(variable)[1:])
    return tf.reshape(variable, shape=[-1, dim], name=name)


class FlattenLayerCustomized(Layer):
    """A COPY FROM THE TENSORLAYER SRC.
    The :class:`FlattenLayer` class is layer which reshape high-dimension
    input to a vector. Then we can apply DenseLayer, RNNLayer, ConcatLayer and
    etc on the top of it.

    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2dLayer(network,
                       act = tf.nn.relu,
                       shape = [5, 5, 32, 64],
                       strides=[1, 1, 1, 1],
                       padding='SAME',
                       name ='cnn_layer')
    network = tl.layers.Pool2dLayer(network,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       pool = tf.nn.max_pool,
                       name ='pool_layer',)
    network = tl.layers.FlattenLayer(network, name='flatten_layer')
    """
    def __init__(
            self,
            layer = None,
            name ='flatten_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.outputs = flatten_unknown(self.inputs, name=name)
        # self.n_units = int(self.outputs.get_shape()[-1])
        #self.n_units = tf.to_int32(tf.shape(self.outputs)[-1])
        print("  [TL] FlattenLayer %s: unknown" % (self.name)) #, self.n_units))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )


# Copied from TensorLayer 1.3 API. We didn't want to upgrade.
class StackLayer(Layer):
    """A COPY FROM THE TENSORLAYER SRC.
    The :class:`StackLayer` class is layer for stacking a list of rank-R tensors into one rank-(R+1) tensor, see `tf.stack() <https://www.tensorflow.org/api_docs/python/tf/stack>`_.

    Parameters
    ----------
    layer : a list of :class:`Layer` instances
        The `Layer` class feeding into this layer.
    axis : an int
        Dimension along which to concatenate.
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
            self,
            layer = [],
            axis = 0,
            name ='stack',
    ):
        Layer.__init__(self, name=name)
        self.inputs = []
        for l in layer:
            self.inputs.append(l.outputs)

        self.outputs = tf.stack(self.inputs, axis=axis, name=name)

        print("  [TL] StackLayer %s: axis: %d" % (self.name, axis))

        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)


class ReshapeLayerCustomized(Layer):
    """BASED ON THE TENSORLAYER SRC.
    The :class:`ReshapeLayer` class is layer which reshape the tensor.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    shape : a list
        The output shape.
    name : a string or None
        An optional name to attach to this layer.

    Examples
    --------
    - The core of this layer is ``tf.reshape``.
    - Use TensorFlow only :
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.reshape(x, shape=[-1, 3, 3])
    sess = tf.InteractiveSession()
    print(sess.run(y, feed_dict={x:[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]]}))
    [[[ 1.  1.  1.]
    [ 2.  2.  2.]
    [ 3.  3.  3.]]
    [[ 4.  4.  4.]
    [ 5.  5.  5.]
    [ 6.  6.  6.]]]
    """
    def __init__(
            self,
            layer=None,
            batchsize=16,
            to_shape=[1,1,1,1],
            name='reshape_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        splits = [t for t in tf.split(self.inputs, batchsize, axis=0)]
        reshaped = [tf.reshape(t, to_shape) for t in splits]
        self.outputs = tf.concat(reshaped, axis=0)
        print("  [TL] ReshapeLayerCustomized %s: %s" % (self.name, self.outputs.get_shape()))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        del splits


class PoolLayer1d(Layer):
    """A COPY FROM THE TENSORLAYER SRC.
    The :class:`PoolLayer` class is a Pooling layer, you can choose
    ``tf.nn.max_pool`` and ``tf.nn.avg_pool`` for 2D or
    ``tf.nn.max_pool3d`` and ``tf.nn.avg_pool3d`` for 3D.

    Parameters
    ----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    ksize : a list of ints that has length = 3.
        The size of the window for each dimension of the input tensor.
    strides : a list of ints that has length = 3.
        The stride of the sliding window for each dimension of the input tensor.
    padding : a string from: "SAME", "VALID".
        The type of padding algorithm to use.
    pool : a pooling function
        - see `TensorFlow pooling APIs <https://www.tensorflow.org/versions/master/api_docs/python/nn.html#pooling>`_
        - class ``tf.nn.max_pool1d``
    name : a string or None
        An optional name to attach to this layer.
    """
    def __init__(
            self,
            layer = None,
            ksize = [2],
            strides = [2],
            pooling_type = "MAX",
            padding = 'SAME',
            pool = tf.nn.pool,
            name = 'pool_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] PoolLayer   %s: ksize:%s strides:%s padding:%s pool:%s" %
              (self.name, str(ksize), str(strides), padding, pool.__name__))

        self.outputs = pool(self.inputs, window_shape=ksize, pooling_type=pooling_type, strides=strides,
                            padding=padding, name=name)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
