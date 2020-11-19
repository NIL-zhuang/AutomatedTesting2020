import keras
from keras.datasets import mnist, cifar100
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def test_preprocess(img: np.ndarray) -> np.ndarray:
    return gaussian_filter(img, sigma=1.2)


testgen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    preprocessing_function=test_preprocess,
)


# dataset = keras.datasets.cifar100.load_data()
model_path = "/home/nil/Documents/code/AutomatedTesting2020/my_dnn_final.hdf5"
model = keras.models.load_model(model_path)
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

testgen.fit(X_test)
test_iter = testgen.flow(X_test, y_test, batch_size=10000)
gen_test_X, gen_test_y = test_iter.next()
# X_train = X_train.reshape(-1, 28, 28, 1)
# X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
print(model.evaluate(X_test.reshape(-1, 784), y_test, verbose=1))
print("=============================")
print(model.evaluate(gen_test_X.reshape(-1, 784), gen_test_y, verbose=1))
