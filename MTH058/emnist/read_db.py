#!/usr/bin/python3

import sys, os, re, random

import numpy as np
from PIL import Image

# Get db_dir
try:
  db_dir = sys.argv[1]
except IndexError:
  print("Usage: " + os.path.basename(__file__) + " <db_dir>")
  sys.exit(1)

# Check if db_dir is a directory
if not os.path.isdir(db_dir):
  raise ValueError(db_dir + "is not a directory")

# Read db_dir
# Type of db: EMNIST by_field
path_pattern = "{}/{}"
data = {}
file_pattern = re.compile("^\w{8}\.png$")
def add_data(path, label):
  subpath = path_pattern.format(path, label)
  if label not in data:
    data[label] = []
  if os.path.isdir(subpath):
    for i in os.listdir(subpath):
      file_path = path_pattern.format(subpath, i)
      if file_pattern.match(i):
        data[label].append(file_path)
try:
  db_subdir = list(filter(lambda x: re.match("^hsf_.$", x), os.listdir(db_dir)))
  for i in db_subdir:
    path = path_pattern.format(db_dir, i)
    subpath = path_pattern.format(path, "digit")
    if os.path.isdir(subpath):
      for j in range(10):
        label = "{:02x}".format(ord("0") + j)
        add_data(subpath, label)
    subpath = path_pattern.format(path, "lower")
    if os.path.isdir(subpath):
      for j in range(26):
        label = "{:02x}".format(ord("a") + j)
        add_data(subpath, label)
    subpath = path_pattern.format(path, "upper")
    if os.path.isdir(subpath):
      for j in range(26):
        label = "{:02x}".format(ord("A") + j)
        add_data(subpath, label)
except:
  print("Type of db is not 'EMNIST by_field'")
  sys.exit(1)

# Convert to idx file format (MNIST DB format)
# Set train and test ratio
train_ratio = 0.8 # test_ratio = 1 - train_ratio

# Split dataset
train_data = {}
test_data = {}
for i in data:
  train_data[i] = []
  test_data[i] = []
  for j in data[i]:
    if random.random() < train_ratio:
      train_data[i].append(j)
    else:
      test_data[i].append(j)

# Get size of each dataset
def count_dataset(dataset):
  n = 0
  for i in dataset:
    n += len(dataset[i])
  return n
train_number = count_dataset(train_data)
test_number = count_dataset(test_data)

# Create train dataset
def create_dataset(prefix, number, dataset, cnt):
  labels_file = open("{}-labels-idx1-ubyte".format(prefix), "wb")
  images_file = open("{}-images-idx3-ubyte".format(prefix), "wb")
  labels_file.write(b"\x00\x00\x08\x01")
  images_file.write(b"\x00\x00\x08\x03")
  labels_file.write(int.to_bytes(number, length=4, byteorder="big"))
  images_file.write(int.to_bytes(number, length=4, byteorder="big"))
  images_file.write(int.to_bytes(128, length=4, byteorder="big"))
  images_file.write(int.to_bytes(128, length=4, byteorder="big"))
  labels = b""
  for i in dataset:
    for j in dataset[i]:
      print("Processing image #{}/{}".format(cnt, train_number + test_number), end="\r")
      im = np.average(np.array(Image.open(j)), axis=2).astype("uint8")
      if im.shape != (128, 128):
        raise ValueError("Shape of image in file {} is {}".format(j, im.shape))
      images_file.write(im.tobytes())
      labels += bytes.fromhex(i)
      cnt += 1
  labels_file.write(labels)
  labels_file.close()
  images_file.close()
create_dataset("train", train_number, train_data, 1)
create_dataset("test", test_number, test_data, 1 + train_number)

