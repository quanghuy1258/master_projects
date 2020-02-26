#!/usr/bin/python3

from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer

# Reference: http://keras.io/examples/cifar10_cnn_capsule/

def squash(x):
  x_squared_norm = K.sum(K.square(x)) + K.epsilon()
  scale = K.sqrt(x_squared_norm) / (1.0 + x_squared_norm)
  return scale * x

class Capsule(Layer):
  def __init__(self, num_capsule, dim_capsule, routings=3,  activation="squash",**kwargs):
    self.num_capsule = num_capsule
    self.dim_capsule = dim_capsule
    self.routings = routings
    if activation == "squash":
      self.activation = squash
    else:
      self.activation = activations.get(activation)
    super(Capsule, self).__init__(**kwargs)

  def build(self, input_shape):
    super(Capsule, self).build(input_shape)

  def call(self):
    return

  def compute_output_shape(self):
    return
