# Uyen's Read Dataset

import os, cv2, itertools

import numpy as np
from sklearn.model_selection import train_test_split

ROWS = 64
COLS = 64
CHANNELS = 3

# Some necessary functions
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def read_dataset(dataset_dir):
    # Normalize directory's name
    if not dataset_dir.startswith('./'):
        dataset_dir = './' + dataset_dir
    if not dataset_dir.endswith('/'):
        dataset_dir = dataset_dir + '/'

    # Get images' path
    list_path_images = [dataset_dir + i for i in os.listdir(dataset_dir)]

    # Get num images
    num_images = len(list_path_images)

    #X = np.ndarray((m, ROWS, COLS, CHANNELS), dtype=np.uint8)
    X_dog_list = []
    y_dog_list = []
    X_cat_list = []
    y_cat_list = []

    for i, image_file in enumerate(list_path_images):
        image = read_image(image_file)

        x_temp = np.squeeze(image.reshape((ROWS, COLS, CHANNELS)))

        if 'dog' in image_file.lower():
            X_dog_list.append(x_temp)
            y_dog_list.append(1)
        elif 'cat' in image_file.lower():
            X_cat_list.append(x_temp)
            y_cat_list.append(1)
        else:
            print('ERROR DATA')
            
        if i % 5000 == 0 :
            print("Proceed {} of {}".format(i, num_images))

    # Convert to np.array
    X_dog = np.array(X_dog_list, dtype=np.uint8).reshape((-1, ROWS, COLS, CHANNELS))
    y_dog = np.array(y_dog_list, dtype=np.uint8).reshape((-1, 1))
    X_cat = np.array(X_cat_list, dtype=np.uint8).reshape((-1, ROWS, COLS, CHANNELS))
    y_cat = np.array(y_cat_list, dtype=np.uint8).reshape((-1, 1))

    return X_dog, y_dog, X_cat, y_cat


def create_train_test_dataset(dataset_dir):
    X_dog, y_dog, X_cat, y_cat = read_dataset(dataset_dir)

    X_train_dog, X_test_dog, y_train_dog, y_test_dog = train_test_split(X_dog, y_dog, test_size=0.2, random_state=42)
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)

    X_train = np.concatenate((X_train_dog, X_train_cat), axis=0)
    y_train = np.concatenate((y_train_dog, y_train_cat), axis=0)

    X_test = np.concatenate((X_test_dog, X_test_cat), axis=0)
    y_test = np.concatenate((y_test_dog, y_test_cat), axis=0)

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = create_train_test_dataset('train')

print(f'X_train.shape {X_train.shape}')
print(f'y_train.shape {y_train.shape}')
print(f'X_test.shape {X_test.shape}')
print(f'y_test.shape {y_test.shape}')