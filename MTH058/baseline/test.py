#!/usr/bin/python3

import sys, os

import numpy as np

import common

# Load data
prefix = "t10k"
images = common.read_images(prefix)
ori_labels = common.read_labels(prefix)
new_labels = common.category2binary(ori_labels)

# Create model
model = common.create_model()

# Load weights
try:
  checkpoint_filepath = sys.argv[1]
  model.load_weights(checkpoint_filepath)
except IndexError:
  print("Usage: " + os.path.basename(__file__) + " <checkpoint_filepath>")
  sys.exit(1)

# Testing
score = model.evaluate(images, new_labels)
common.print_score(score)

# Get and save result
result = common.binary2category(model.predict(images))
testing_dir = "testing"
filepath = "result.txt"
common.create_dir_if_not_exists(testing_dir)
np.savetxt(testing_dir+"/"+filepath, result, fmt="%d")
