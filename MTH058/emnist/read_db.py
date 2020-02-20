#!/usr/bin/python3

import sys, os, re

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

with open("index_db.txt", "w") as f:
  for i in data:
    for j in data[i]:
      f.write("{} {}\n".format(i, j))
