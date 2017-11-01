import tensorflow as tf
import tensorlayer as tl
import math
from tensorflow.contrib.losses.python.losses.loss_ops import *
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.deprecation import deprecated


def focal_lossIII(prediction_tensor,
                  target_tensor,
                  weights,
                  gamma=2.,
                  epsilon=0.00001,
                  ):
    """Compute loss function.
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
    pos_per_seq = tf.reduce_sum(target_tensor, axis=1)
    neg_per_seq = tf.reduce_sum(1. - target_tensor, axis=1)

    #ratio = tf.expand_dims(tf.divide(pos_per_seq, neg_per_seq), axis=-1)

    #weights = tf.expand_dims(weights, 0)
    preds = tf.nn.sigmoid(prediction_tensor)
    preds = tf.where(tf.equal(target_tensor, 1), preds, 1. - preds)
    #pos_losses = - tf.log(preds + epsilon) * target_tensor
    #neg_losses = - tf.log(preds + epsilon) * (1. - target_tensor) * ratio
    losses = (1. - preds) ** gamma * tf.nn.sigmoid_cross_entropy_with_logits(labels=target_tensor,
                                                                              logits=prediction_tensor)

    #return tf.multiply(losses, weights)
    return losses


def focal_loss(labels=[], logits=[], pos_weights=[], gamma=2., clips=[], name='focal_loss'):
    """
    Add focal loss weigths to the wigthted sigmoid cross entropy
    :return:
    """
    batchsize = labels.get_shape().as_list()[0]
    n_classes = labels.get_shape().as_list()[1]

    with tf.variable_scope(name) as vs:
        # first get a sigmoid to determine the focal loss weigths:
        sigmoid_logits = tf.nn.sigmoid(logits)
        # determine the focal loss weigths:
        labels = math_ops.to_float(labels)
        sigmoid_logits.get_shape().assert_is_compatible_with(labels.get_shape())
        preds = array_ops.where(math_ops.equal(labels, 1.), sigmoid_logits, 1. - sigmoid_logits)
        focal_weights = (math_ops.subtract(1., preds)) ** gamma
        print(focal_weights)

        # clip the weights at E-3 and E3
        up_clip = math_ops.multiply(tf.ones([batchsize, n_classes]), clips[1])
        low_clip = math_ops.multiply(tf.ones([batchsize, n_classes]), clips[0])
        focal_weights = array_ops.where(math_ops.greater(focal_weights, clips[1]), up_clip, focal_weights)
        focal_weights = array_ops.where(math_ops.less(focal_weights, clips[0]), low_clip, focal_weights)
        log_weight = 1. + (pos_weights - 1.) * labels

        # now put them into a weighted softmax ce:
        loss = math_ops.multiply(math_ops.add((1. - labels) * logits,
                        log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) + nn_ops.relu(-logits))),
               focal_weights, name='sc_entropy')
        return loss

def focal_loss_alpha(labels=[], logits=[], pos_weights=[], gamma=2., clips=[], name='focal_loss'):
    """
    Add focal loss weigths to the wigthted sigmoid cross entropy
    :return:
    """
    batchsize = labels.get_shape().as_list()[0]
    n_classes = labels.get_shape().as_list()[1]

    with tf.variable_scope(name) as vs:
        # first get a sigmoid to determine the focal loss weigths:
        sigmoid_logits = tf.nn.sigmoid(logits)
        # determine the focal loss weigths:
        labels = math_ops.to_float(labels)
        sigmoid_logits.get_shape().assert_is_compatible_with(labels.get_shape())
        preds = array_ops.where(math_ops.equal(labels, 1.), sigmoid_logits, 1. - sigmoid_logits)
        focal_weights = (math_ops.subtract(1., preds)) ** gamma
        print(focal_weights)

        # clip the weights at E-3 and E3
        up_clip = math_ops.multiply(tf.ones([batchsize, n_classes]), clips[1])
        low_clip = math_ops.multiply(tf.ones([batchsize, n_classes]), clips[0])
        focal_weights = array_ops.where(math_ops.greater(focal_weights, clips[1]), up_clip, focal_weights)
        focal_weights = array_ops.where(math_ops.less(focal_weights, clips[0]), low_clip, focal_weights)
        log_weight = 1. + (pos_weights - 1.) * labels

        # now put them into a weighted softmax ce:
        loss = math_ops.multiply(math_ops.add((1. - labels) * logits,
                         log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) + nn_ops.relu(-logits))),
                                 focal_weights, name='sc_entropy')
        return loss

def resnet_block(inlayer, channels=[128, 256], pool_dim=2, is_train=True, name='scope'):
    """
    Define a resnet block for DeeProtein
    :param in_channels:
    :param out_channels:
    :param name:
    :return:
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
    v = tf.get_variable("prelu_weight", dtype=tf.float32, initializer=tf.constant(0.2))
    return tf.where(tf.greater_equal(x, 0), x, x*v)


def binary_focal_loss(predictions, labels, gamma=2, alpha=1, weights=1.0, epsilon=1e-7, scope=None):
    """Adds a Focal Loss term to the training procedure.

    For each value x in `predictions`, and the corresponding l in `labels`,
    the following is calculated:

    ```
      pt = 1 - x                  if l == 0
      pt = x                      if l == 1

      focal_loss = - a * (1 - pt)**g * log(pt)
    ```

    where g is `gamma`, a is `alpha`.

    See: https://arxiv.org/pdf/1708.02002.pdf

    `weights` acts as a coefficient for the loss. If a scalar is provided, then
    the loss is simply scaled by the given value. If `weights` is a tensor of size
    [batch_size], then the total loss for each sample of the batch is rescaled
    by the corresponding element in the `weights` vector. If the shape of
    `weights` matches the shape of `predictions`, then the loss of each
    measurable element of `predictions` is scaled by the corresponding value of
    `weights`.

    Args:
      labels: The ground truth output tensor, same dimensions as 'predictions'.
      predictions: The predicted outputs.
      gamma, alpha: parameters.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `losses` dimension).
      epsilon: A small increment to add to avoid taking a log of zero.
      scope: The scope for the operations performed in computing the loss.
      loss_collection: collection to which the loss will be added.
      reduction: Type of reduction to apply to loss.

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
      shape as `labels`; otherwise, it is scalar.

    Raises:
      ValueError: If the shape of `predictions` doesn't match that of `labels` or
        if the shape of `weights` is invalid.
    """
    with ops.name_scope(scope, "focal_loss",
                        (predictions, labels, weights)) as scope:
        predictions = math_ops.to_float(predictions)
        labels = math_ops.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        preds = array_ops.where(
            math_ops.equal(labels, 1), predictions, 1. - predictions)
        losses = -alpha * (1. - preds)**gamma * math_ops.log(preds + epsilon)
        return compute_weighted_loss(losses, weights, scope=scope)


Layer = tl.layers.Layer


def list_remove_repeat(l=None):
    """Remove the repeated items in a list, and return the processed list.
    You may need it to create merged layer like Concat, Elementwise and etc.

    Parameters
    ----------
    l : a list

    Examples
    ---------
    """
    l2 = []
    [l2.append(i) for i in l if not i in l2]
    return l2


class PadLayer(Layer):
    """
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
    """Reshapes high-dimension input to a vector.
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
    """
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

class Pseudo1dDeconv(Layer):
    """
    Pseudo 1d deconv layer. We apply padding on the input to exactly fit the kernel without setting the
    padding in the deconv layer to 'SAME'. We do not want to pad in the deconv as we want to keep the width dimension constant
    """
    def __init__(
            self,
            layer=None,
            name='x'
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        # TODO: finish this.
        self.outputs = 0
        #print("  [TL] ReshapeLayerCustomized %s: %s" % (self.name, self.outputs.get_shape()))
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])

def Pseudo1dDeconvBundle(input, n_out=128, kernel_size=[3, 3], name='deconv'):
    """
    Pseudo 1d deconv layer. We apply padding on the input to exactly fit the kernel without setting the
    padding in the deconv layer to 'SAME'. We do not want to pad in the deconv as we want to keep the width dimension constant

    - Kernel_height has to equal to the inout AND output sizes
    - Kernel_width has to be an odd number.

    :param input:
    :param name:
    :param input_shape: [height, width] 2d tensor
    :param kernel_size: [height, width]
    :return:
    """
    with tf.variable_scope(name) as vs:
        input_shape = input.get_shape().as_list() # should be shape [batchsize, height, width, channels]

        assert kernel_size[1] % 2 != 0
        assert kernel_size[0] == input_shape[1]

        padding_pos = math.floor(kernel_size[1])

        paddings = [[0, 0], [0, 0], [padding_pos, padding_pos], [0, 0]]

        pad_input = PadLayer(input, paddings=paddings, mode='CONSTANT', name='pad_layer')

        deconv = tl.layers.DeConv2dLayer(pad_input,
                                         act=tf.nn.relu, # TODO: make PRELU great again?
                                         # shape is selected such that the length dim stays the same:
                                         # This means:
                                         shape=[kernel_size[0], kernel_size[1], n_out, input_shape[-1]],
                                         output_shape=[input_shape[0], input_shape[1], input_shape[2]-2*padding_pos, n_out],
                                         # TODO: compare this with the 1d output
                                         #output_shape=[input_shape[0], input_shape[1], 1, n_out],
                                         strides=[1, 5, 1, 1],
                                         padding='VALID',
                                         W_init=tf.truncated_normal_initializer(stddev=0.02), #TODO check if this is too much
                                         b_init=tf.constant_initializer(value=0.01),
                                         W_init_args={}, b_init_args={},
                                         name='decnn2d_layer')
    return deconv


def DeConv2d(net, n_out_channel = 32, filter_size=(3, 3),
        out_size = (30, 30), strides = (2, 2), padding = 'SAME', batch_size = None, act = None,
        W_init = tf.truncated_normal_initializer(stddev=0.02), b_init = tf.constant_initializer(value=0.0),
        W_init_args = {}, b_init_args = {}, name ='decnn2d'):
    """Wrapper for :class:`DeConv2dLayer`, if you don't understand how to use :class:`DeConv2dLayer`, this function may be easier.

    Parameters
    ----------
    net : TensorLayer layer.
    n_out_channel : int, number of output channel.
    filter_size : tuple of (height, width) for filter size.
    out_size :  tuple of (height, width) of output.
    batch_size : int or None, batch_size. If None, try to find the batch_size from the first dim of net.outputs (you should tell the batch_size when define the input placeholder).
    strides : tuple of (height, width) for strides.
    act : None or activation function.
    others : see :class:`DeConv2dLayer`.
    """
    assert len(strides) == 2, "len(strides) should be 2, DeConv2d and DeConv2dLayer are different."
    if act is None:
        act = tf.identity
    if batch_size is None:
    #     batch_size = tf.shape(net.outputs)[0]
        fixed_batch_size = net.outputs.get_shape().with_rank_at_least(1)[0]
        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = tf.array_ops.shape(net.outputs)[0]
    net = tl.layers.DeConv2dLayer(layer = net,
                    act = act,
                    shape = [filter_size[0], filter_size[1], n_out_channel, int(net.outputs.get_shape()[-1])],
                    output_shape = [batch_size, int(out_size[0]), int(out_size[1]), n_out_channel],
                    strides = [1, strides[0], strides[1], 1],
                    padding = padding,
                    W_init = W_init,
                    b_init = b_init,
                    W_init_args = W_init_args,
                    b_init_args = b_init_args,
                    name = name)
    return net

# Copied from TensorLayer 1.3 API. We didn't want to upgrade.
class StackLayer(Layer):
    """
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
    """
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
    """
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


class SpatialPyramidPoolinglayer1d(Layer):
    def spp(self, sequences, lengths, levels, divsPerLevel):
        with tf.name_scope("spp"):
            batch_size, _, channels = sequences.get_shape()
            batchValues = []
            # we cannot max_pool2d because sequences have different lengths
            for b in range(batch_size):
                currLength = lengths[b]
                pooledValues = []
                for level in range(levels):
                    ndiv = divsPerLevel ** level
                    assert ndiv > 0
                    divLength = tf.cast(tf.ceil(tf.truediv(currLength, ndiv) - 1e-8), tf.int32)
                    for divIndex in range(ndiv):
                        divStart = 0 if ndiv <= 1 else tf.cast(
                            tf.round(tf.truediv(currLength - divLength, ndiv - 1) * divIndex), tf.int32)
                        pooledValues.append(tf.cond(tf.greater(divLength, 0), lambda: tf.reduce_max(
                            sequences[b, divStart:divStart + divLength, :], 0),
                                                    lambda: tf.zeros(shape=[channels], dtype=tf.float32)))
                spp_count = len(pooledValues)
                pooledValue = tf.stack(pooledValues, 0)
                pooledValue.set_shape([spp_count, channels])
                batchValues.append(pooledValue)
            result = tf.stack(batchValues, 0)
            result.set_shape([batch_size, spp_count, channels])
        return result

    def __init__(self,
                 layer = None,
                 lengths = None,
                 pool_lvls=3,
                 pool_divs=4,
                 mode='max',
                 name = 'SPPlayer'
                 ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        self.lengths = lengths
        self.mode = mode

        # counter and container
        self.results = []
        self.out_dim = 0

        print("  [TL] SPPLayer   %s: pool_lvls:%s pool_divs:%s mode:%s" %
              (self.name, str(pool_lvls), str(pool_divs), mode))

        self.outputs = self.spp(self.inputs, self.lengths, pool_lvls, pool_divs)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )

