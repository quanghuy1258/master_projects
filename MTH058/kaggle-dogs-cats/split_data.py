import os
from shutil import copyfile

import random


dataset_dir = "./train/"
list_path_images = [dataset_dir + i for i in os.listdir(dataset_dir)]

set_path_cat_images = set()
set_path_dog_images = set()

for path in list_path_images:
    if 'dog' in path.lower():
        set_path_dog_images.add(path)
    elif 'cat' in path.lower():
        set_path_cat_images.add(path)
        
print('Number of dog files: ', len(set_path_dog_images))
print('Number of cat files: ', len(set_path_cat_images))

# Choose sample to test
test_cat_images = set(random.sample(set_path_cat_images, 2500))
test_dog_images = set(random.sample(set_path_dog_images, 2500))

train_cat_images = set_path_cat_images - test_cat_images
train_dog_images = set_path_dog_images - test_dog_images

print('Number of dog files in train dir: ', len(train_dog_images))
print('Number of cat files in train dir: ', len(train_cat_images))
print('Number of dog files in test dir: ', len(test_dog_images))
print('Number of cat files in test dir: ', len(test_cat_images))


def create_dataset_dir(dir_name, list_path_files):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
        
    for path in list_path_files:
        copyfile(path, dir_name + '/' + path.split('/')[-1]) 
    
create_dataset_dir('test_new', test_cat_images | test_dog_images)
create_dataset_dir('train_new', train_cat_images | train_dog_images)