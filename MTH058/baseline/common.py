#!/usr/bin/python3

import numpy as np

def read_bin_file(name):
  with open(name, "rb") as f:
    return f.read()

def read_images(prefix):
  name = prefix + "-images-idx3-ubyte"
  data = read_bin_file(name)
  num = int.from_bytes(data[4:8], "big")
  images = np.frombuffer(data, dtype=np.uint8, offset=16).reshape([num, 28 * 28])
  return images

def read_labels(prefix):
  name = prefix + "-labels-idx1-ubyte"
  data = read_bin_file(name)
  num = int.from_bytes(data[4:8], "big")
  labels = np.frombuffer(data, dtype=np.uint8, offset=8)
  return labels

