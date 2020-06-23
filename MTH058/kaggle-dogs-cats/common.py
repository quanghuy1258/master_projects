import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop, SGD

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from tensorflow.keras.utils import to_categorical

def create_dir_if_not_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def create_checkpoint_callback(path, period):
    return ModelCheckpoint(filepath=path, verbose=1, save_weights_only=True, period=period)

def create_CSVLogger_callback(training_dir):
    return CSVLogger("{}/{}".format(training_dir, "log.csv"))

def create_baseline_model():
    model = Sequential()
  
    model.add(Dense(512, activation='relu', input_shape=(64, 64, 3)))
    model.add(Dense(256, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def create_lenet5_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

#     opt = SGD(lr=0.001, momentum=0.9)
#     model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_cnn_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(filters=512, kernel_size=(1,1), activation='relu'))

    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_alexnet_model():
    model = Sequential()
    
    # Alexnet
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    model.add(Dropout(0.4))
    
    # 1st Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.2))

    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation='relu'))

    # 3rd Fully Connected Layer
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    
#     opt = SGD(lr=0.001, momentum=0.9)
#     model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_vgg_model():
    model = Sequential()

    # VGG16
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(64, 64, 3)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
              
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
              
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
              
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
              
    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.2))

    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 3rd Fully Connected Layer
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
              
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def visualize_history(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Categorical Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.show()