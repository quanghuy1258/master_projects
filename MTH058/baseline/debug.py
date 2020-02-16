#!/usr/bin/python3

import sys, os

import numpy as np

import common

# Load data
prefix = "t10k"
images = common.read_images(prefix)
labels = common.read_labels(prefix)

# Load result
try:
  result_filepath = sys.argv[1]
  result = np.loadtxt(result_filepath).astype("float32")
  result = result.reshape([result.shape[0], 1])
except IndexError:
  print("Usage: " + os.path.basename(__file__) + " <result_filepath>")
  sys.exit(1)

# Get false indices
false_indices = np.argwhere(labels != result)[:, 0]

# Debugging
debug_dir = "debugging"
common.create_dir_if_not_exists(debug_dir)
for i in false_indices:
  common.debug(debug_dir, i, images[i], labels[i][0], result[i][0])
