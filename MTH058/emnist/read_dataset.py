import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import keras

dataTrain = pd.read_csv("data/emnist-balanced-train.csv")
dataTest = pd.read_csv("data/emnist-balanced-test.csv")

# train, validate = train_test_split(dataTrain, test_size=0.1) # change this split however you want



def prepare_data_for_resnet50(data_to_transform):
    data = data_to_transform.copy().values
    data = data[:, 1:]
    data = data.reshape(-1, 28, 28) / 255
    data = X_rgb = np.stack([data, data, data], axis=-1)
    return data



def getTrainData():
    X = prepare_data_for_resnet50(dataTrain)
    y = dataTrain.copy().values[:, 0:1]
    y = keras.utils.to_categorical(y, 47)
    print(X.shape, y.shape)
    return X, y

def getTestData():
    X = prepare_data_for_resnet50(dataTest)
    y = dataTest.copy().values[:, 0:1]
    y = keras.utils.to_categorical(y, 47)
    print(X.shape, y.shape)
    return X, y
