#!/usr/bin/python3

import common

prefix = "train"
images = common.read_images(prefix)
labels = common.read_labels(prefix)

import tensorflow as tf
