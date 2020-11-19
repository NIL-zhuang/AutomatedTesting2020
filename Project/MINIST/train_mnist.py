import keras
from keras import models
from keras.callbacks import TensorBoard, EarlyStopping
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import numpy as np
import random
from os import listdir
from scipy.ndimage.filters import gaussian_filter
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.optimizers import Optimizer


norm = [-0.38, -0.37, -0.35, -0.33, -0.3, -0.28, -0.25,
        -0.23, -0.2, -0.17, -0.13, -0.1, -0.07, -0.03,
        0.0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2,
        0.23, 0.25, 0.28, 0.3, 0.33, 0.35, 0.37]


def test_preprocess(img: np.ndarray) -> np.ndarray:
    return gaussian_filter(img, sigma=1.2)


def train_preprocess(img: np.ndarray) -> np.ndarray:
    return wrap(img)


def wrap(img: np.ndarray, alpha=1.2):
    n_img = np.full([28, 28, 1], 254.)
    alpha = alpha * random.random()
    for i in range(28):
        for j in range(28):
            offset_x = int(alpha * (norm[i]-0.5))
            offset_y = int(alpha * (norm[j]-0.5))
            if i+offset_x < 28 and j+offset_y < 28:
                n_img[i, j] = img[(i+offset_x), (j+offset_y)]
    return n_img


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


traingen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    preprocessing_function=train_preprocess,
    rotation_range=15,
    fill_mode='nearest'
)
traingen.fit(X_train)
testgen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    preprocessing_function=test_preprocess,
)
testgen.fit(X_test)

train_iter = traingen.flow(X_train, y_train, batch_size=400000)
test_iter = testgen.flow(X_test, y_test, batch_size=20000)
gen_train_X, gen_train_y = train_iter.next()
gen_test_X, gen_test_y = test_iter.next()
model_path = "/home/nil/Documents/code/AutomatedTesting2020/Models/MNIST/dnn_without_dropout.hdf5"
model = keras.models.load_model(model_path)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(gen_train_X.reshape(-1, 784), gen_train_y, batch_size=32, epochs=70, shuffle=True,
          callbacks=[EarlyStopping(monitor='loss', patience=5), TensorBoard(log_dir='./logs')])
model.save("my_dnn_final_2.hdf5")
print(model.evaluate(gen_test_X.reshape(-1, 784), gen_test_y, verbose=1))
print(model.evaluate(X_test.reshape(-1, 784), y_test, verbose=1))
