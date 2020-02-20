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
data = {"cat": [], "dog": []}
cat_pattern = re.compile("^cat\..+\.jpg$")
dog_pattern = re.compile("^dog\..+\.jpg$")
try:
  for i in os.listdir(db_dir):
    if cat_pattern.match(i):
      data["cat"].append(path_pattern.format(db_dir, i))
    if dog_pattern.match(i):
      data["dog"].append(path_pattern.format(db_dir, i))
except:
  print("db is not 'dogs-vs-cats'")
  sys.exit(1)

with open("index_db.txt", "w") as f:
  for i in data:
    for j in data[i]:
      f.write("{} {}\n".format(i, j))
