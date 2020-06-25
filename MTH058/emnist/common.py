import os

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

def create_dir_if_not_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def create_checkpoint_callback(path, period):
    return ModelCheckpoint(filepath=path, verbose=1, save_weights_only=True, period=period)

def create_CSVLogger_callback(training_dir):
    return CSVLogger("{}/{}".format(training_dir, "log.csv"))

# def create_cnn_model():