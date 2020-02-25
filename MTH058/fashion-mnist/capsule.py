#!/usr/bin/python3

from keras import backend as K
from tensorflow.keras.layers import Layer

# Reference: http://keras.io/examples/cifar10_cnn_capsule/
class Capsule(Layer):
  def __init__(self, num_capsule, dim_capsule, **kwargs):
    self.num_capsule = num_capsule
    self.dim_capsule = dim_capsule
    super(Capsule, self).__init__(**kwargs)

  def build(self, input_shape):
    super(Capsule, self).build(input_shape)

  def call(self):
    return

  def compute_output_shape(self):
    return
