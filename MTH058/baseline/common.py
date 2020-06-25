#!/usr/bin/python3

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

# Metadata
input_size = 28 * 28
output_size = 10
input_layer = input_size
layer_1 = 500
layer_2 = 150
output_layer = output_size

def create_dir_if_not_exists(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def read_bin_file(name):
  with open(name, "rb") as f:
    return f.read()

def read_images(prefix):
  name = prefix + "-images-idx3-ubyte"
  data = read_bin_file(name)
  num = int.from_bytes(data[4:8], "big")
  images = np.frombuffer(data, dtype=np.uint8, offset=16).reshape([num, input_size]).astype("float32")
  return images

def read_labels(prefix):
  name = prefix + "-labels-idx1-ubyte"
  data = read_bin_file(name)
  num = int.from_bytes(data[4:8], "big")
  labels = np.frombuffer(data, dtype=np.uint8, offset=8).reshape([num, 1]).astype("float32")
  return labels

def category2binary(x):
  temp = [None] * output_size
  for i in range(output_size):
    temp[i] = np.zeros(output_size)
    temp[i][i] = 1
  return np.apply_along_axis(lambda a: temp[int(a[0])], 1, x).astype("float32")

def binary2category(x):
  return np.apply_along_axis(lambda a: np.argmax(a), 1, x).reshape([x.shape[0], 1]).astype("float32")

def create_checkpoint_callback(path, period):
  return ModelCheckpoint(filepath=path, verbose=1, save_weights_only=True, period=period)

def create_CSVLogger_callback(training_dir):
  return CSVLogger("{}/{}".format(training_dir, "log.csv"))

def create_model():
  model = Sequential()
  model.add(Dense(layer_1, activation="sigmoid", input_shape=(input_layer,)))
  model.add(Dense(layer_2, activation="sigmoid"))
  model.add(Dense(output_layer, activation="softmax"))
  model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
  return model

def visualize_history(history):
  plt.plot(history.history["categorical_accuracy"])
  plt.plot(history.history["val_categorical_accuracy"])
  plt.title("Model accuracy")
  plt.ylabel("Categorical Accuracy")
  plt.xlabel("Epoch")
  plt.legend(["Train", "Validation"])
  plt.show()

def print_score(score):
  print("Loss: {}".format(score[0]))
  print("Categorical Accuracy:: {}".format(score[1]))

def debug(debug_dir, index, image, label, predict):
  im = Image.fromarray(image.reshape(28, 28).astype("uint8"))
  im.save("{}/index_{}_label_{}_predict_{}.png".format(debug_dir, index, int(label), int(predict)))

