import keras
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
