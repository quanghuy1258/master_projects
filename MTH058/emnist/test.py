import sys, os
import numpy as np
import resnet
import densenet
import cv2
import random

import common
import read_dataset as reader

# model = resnet.ResNet50()


X, y = reader.getTestData()

def Eval(model, X, y, checkpoint_filepath):
    try:
        model.load_weights(checkpoint_filepath)
    except IndexError:
        print("Can not load weights from this checkpoint")
        sys.exit(1)
    results = model.evaluate(X, y)
    print(f'Test loss = {results[0]:.2f}')
    print(f'Test acc = {results[1]*100:.2f}%')
    print("=" * 80)
    
def Pred(model, X, y, checkpoint_filepath):
    try:
        model.load_weights(checkpoint_filepath)
    except IndexError:
        print("Can not load weights from this checkpoint")
        sys.exit(1)
    y_predict = model.predict(X)
    rClasses = np.argmax(y,axis=1)
    pClasses = np.argmax(y_predict,axis=1)
    wIDs = []
    rIDs = []
    for i in range(len(rClasses)):
        if rClasses[i]!=pClasses[i]:
            wIDs.append(i)
        else:
            rIDs.append(i)
    wID = random.choice(wIDs)
    rID = random.choice(rIDs)
    wImg = X[wID].reshape((28,28,3))
    cv2.imshow("Random wrong case, index " + wID,wImg)
    rImg = X[rID].reshape((28,28,3))
    cv2.imshow("Random wrong case, index " + rID ,rImg)
    cv2.waitKey()

Eval(densenet.densenet(),X,y,"densenet_training/weights.0060-0.44.h5")
Eval(resnet.ResNet50(),X,y,"training/weights.0030-0.37.h5")
Eval(resnet.ResNet50(),X,y,"training_dropout_lr/weights.0065-0.45.h5")

# Evaluate
# print("=" * 80)
# print('Evaluate on test data')
# # 
# y_predict = model.predict(X)
# img = X[17847].reshape((28,28,3))

# print(t0[17847])

# print(t1[17847])
# cv2.imshow("wrong ",img)
# cv2.waitKey()



#         print(i)


# print(f'Test loss = {results[0]:.2f}')
# print(f'Test acc = {results[1]*100:.2f}%')
# print("=" * 80)