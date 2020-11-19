from random import shuffle
from re import VERBOSE, split
import keras
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import numpy as np
import random
from os import listdir


def show_img():
    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
        # create a grid of 3x3 images
        for i in range(0, 9):
            plt.subplot(330 + 1 + i)
            plt.title(y_batch[i])
            plt.axis('off')
            plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        # show the plot
        plt.show()
        break


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# # define data preparation
# test_datagen = ImageDataGenerator()
# # fit parameters from data
# test_datagen.fit(X_test)
# test_iter = test_datagen.flow(X_test, y_test, batch_size=10000)
# gen_test_X, gen_test_y = test_iter.next()
# validation_image_generator = datagen.flow(X_train, y_train, batch_size=100)

# train_datagen = ImageDataGenerator().fit(X_train)
# train_iter = train_datagen.flow(X_train, y_train, batch_size=50000)
# gen_train_X, gen_train_y = train_iter.next()

Models = "/home/nil/Documents/code/AutomatedTesting2020/Models/MNIST/"

train_RES = []
for model_name in listdir(Models):
    if(model_name.split('_')[0] in ["lenet5"]):
        continue
    print(model_name)
    model_path = Models+model_name
    model = keras.models.load_model(model_path)
    score = None
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    if(model_name.split('_')[0] in ["vgg16", "random1", "random2"]):
        model.fit(X_train, y_train, verbose=2, batch_size=64, shuffle=True, epochs=10, validation_split=0.1)
        score = model.evaluate(X_test, y_test, verbose=1)
    if (model_name.split('_')[0] in ["dnn"]):
        model.fit(X_train.reshape(-1, 784), y_train, verbose=2, batch_size=64, shuffle=True, epochs=10, validation_split=0.1)
        score = model.evaluate(X_test.reshape(-1, 784), y_test, verbose=1)
    train_RES.append([model_name, str(round(score[1], 4))])
print('\n'.join([' '.join(s) for s in train_RES]))
