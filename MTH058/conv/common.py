#!/usr/bin/python3

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

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
  images = np.frombuffer(data, dtype=np.uint8, offset=16).reshape([num, 28, 28, 1]).astype("float32")
  return images

def read_labels(prefix):
  name = prefix + "-labels-idx1-ubyte"
  data = read_bin_file(name)
  num = int.from_bytes(data[4:8], "big")
  labels = np.frombuffer(data, dtype=np.uint8, offset=8).reshape([num, 1]).astype("float32")
  return labels

def category2binary(x):
  temp = [None] * 10
  for i in range(10):
    temp[i] = np.zeros(10)
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
  model.add(Conv2D(50, 3,  activation="sigmoid", input_shape=(28, 28, 1)))
  model.add(Conv2D(50, 3,  activation="sigmoid"))
  model.add(MaxPooling2D())
  model.add(Dropout(0.5))
  model.add(Conv2D(100, 3, activation="sigmoid"))
  model.add(Conv2D(100, 3, activation="sigmoid"))
  model.add(MaxPooling2D())
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(500, activation="sigmoid"))
  model.add(Dense(150, activation="sigmoid"))
  model.add(Dense(10, activation="softmax"))
  model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
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

