#!/usr/bin/python3

# Reference:
# https://arxiv.org/pdf/1710.09829.pdf
# https://keras.io/layers/writing-your-own-keras-layers/
# https://keras.io/examples/cifar10_cnn_capsule/
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras

from tensorflow.keras import initializers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Lambda

def normalize_tuple(value, n, name):
  if isinstance(value, int):
    return (value,) * n
  else:
    try:
      value_tuple = tuple(value)
    except TypeError:
      raise ValueError("The `" + name + "` argument must be a tuple of " +
                       str(n) + " integers. Received: " + str(value))
    if len(value_tuple) != n:
      raise ValueError("The `" + name + "` argument must be a tuple of " +
                       str(n) + " integers. Received: " + str(value))
    for single_value in value_tuple:
      try:
        int(single_value)
      except (ValueError, TypeError):
        raise ValueError("The `" + name + "` argument must be a tuple of " +
                         str(n) + " integers. Received: " + str(value) + " "
                         "including element " + str(single_value) + " of type" +
                         " " + str(type(single_value)))
    return value_tuple

def squash(x, axis=-1):
  x_squared_norm = K.sum(K.square(x), axis=axis, keepdims=True) + K.epsilon()
  scale = K.sqrt(x_squared_norm) / (0.5 + x_squared_norm)
  return scale * x

def margin_loss(y_true, y_pred):
  lamb, margin = 0.5, 0.1
  return K.sum(y_true * K.square(K.relu(1.0 - margin - y_pred)) + lamb * (1.0 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

def OutputLayer():
  return Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=-1)))

class PrimaryCaps2D(Layer):
  """
  # Input shape
      4D tensor with shape:
      `(batch, rows, cols, channels)`
  # Output shape
      3D tensor with shape:
      `(batch, num_capsule, dim_capsule)`
  """
  def __init__(self, channels, dim_capsule, kernel_size, strides=(1, 1), kernel_initializer="glorot_uniform", **kwargs):
    self.out_channels = channels
    self.dim_capsule = dim_capsule
    self.kernel_size = normalize_tuple(kernel_size, 2, "kernel_size")
    self.strides = normalize_tuple(strides, 2, "strides")
    self.kernel_initializer = initializers.get(kernel_initializer)
    super(PrimaryCaps2D, self).__init__(**kwargs)

  def build(self, input_shape):
    if len(input_shape) != 4:
      raise ValueError("Inputs to `PrimaryCaps2D` should have rank 4. "
                       "Received input shape:", str(input_shape))
    inp_channels = input_shape[3]
    kernel_shape = self.kernel_size + (inp_channels, self.out_channels * self.dim_capsule)
    self.kernel = self.add_weight(name="primary_caps2d_kernel",
                                  shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  trainable=True)
    super(PrimaryCaps2D, self).build(input_shape)

  def call(self, inputs):
    inputs = K.conv2d(inputs, self.kernel, self.strides)
    self.num_capsule = inputs.shape[1] * inputs.shape[2] * self.out_channels
    inputs = K.reshape(inputs, (-1, self.num_capsule, self.dim_capsule))
    inputs = squash(inputs)
    return inputs

  def compute_output_shape(self, input_shape):
    return (None, self.num_capsule, self.dim_capsule)

class Capsule(Layer):
  """
  # Input shape
      3D tensor with shape:
      `(batch, inp_num_capsule, inp_dim_capsule)`
  # Output shape
      3D tensor with shape:
      `(batch, out_num_capsule, out_dim_capsule)`
  """
  def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer="glorot_uniform", **kwargs):
    self.num_capsule = num_capsule
    self.dim_capsule = dim_capsule
    self.routings = routings
    self.kernel_initializer = initializers.get(kernel_initializer)
    super(Capsule, self).__init__(**kwargs)

  def build(self, input_shape):
    if len(input_shape) != 3:
      raise ValueError("Inputs to `Capsule` should have rank 3. "
                       "Received input shape:", str(input_shape))
    self.inp_num_capsule = input_shape[1]
    self.inp_dim_capsule = input_shape[2]
    self.kernel = self.add_weight(name="capsule_kernel",
                                  shape=(self.inp_num_capsule, self.inp_dim_capsule, self.num_capsule * self.dim_capsule),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
    super(Capsule, self).build(input_shape)

  def call(self, inputs):
    hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])
    hat_inputs = K.reshape(hat_inputs,
                           (-1, self.inp_num_capsule, self.num_capsule, self.dim_capsule))
    hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
    hat_inputs = K.reshape(hat_inputs,
                           (-1, self.inp_num_capsule, self.dim_capsule))

    b = K.zeros_like(hat_inputs[:, :, 0])
    for i in range(self.routings):
      c = K.reshape(b, (-1, self.num_capsule, self.inp_num_capsule))
      c = K.softmax(c, 1)
      c = K.reshape(c, (-1, self.inp_num_capsule))
      v = squash(K.batch_dot(c, hat_inputs, [1, 1]))
      if i < self.routings - 1:
        b += K.batch_dot(v, hat_inputs, [1, 2])
    return K.reshape(v, (-1, self.num_capsule, self.dim_capsule))

  def compute_output_shape(self):
    return (None, self.num_capsule, self.dim_capsule)

