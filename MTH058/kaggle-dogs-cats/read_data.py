from os import listdir
from numpy import asarray
from numpy import save
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def read_data(dir_name):
    print('\nReading dataset...')
    
    photos, labels = list(), list()
    
    list_path_images = listdir(dir_name)
    
    # Get num images
    num_images = len(list_path_images)

    for i, path in enumerate(list_path_images):
        # Determine class
        output = 0.0
        if path.startswith('cat'):
            output = 1.0
            
        # Load image
        photo = load_img(dir_name + path, target_size=(64, 64))
        
        # Convert to numpy array
        photo = img_to_array(photo)
        
        # Store to list
        photos.append(photo)
        labels.append(output)
        
        if i % 5000 == 0 :
            print("Proceed {} of {}".format(i, num_images))

    # Convert to a numpy arrays
    X = asarray(photos)
    y = asarray(labels)
    
    return X, y


# X_test, y_test = read_data('test_new/')
# print(f'X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}')

# X_train, y_train = read_data('train_new/')
# print(f'X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}')