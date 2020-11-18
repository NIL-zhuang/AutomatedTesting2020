import keras
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import numpy as np


norm = [-0.38, -0.37, -0.35, -0.33, -0.3, -0.28, -0.25,
        -0.23, -0.2, -0.17, -0.13, -0.1, -0.07, -0.03,
        0.0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2,
        0.23, 0.25, 0.28, 0.3, 0.33, 0.35, 0.37]


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    @param img 一个秩为3个Numpy张量，输出同尺寸
    """
    img = pepper_noise(img)
    return wrap(img)
    # return img


def pepper_noise(img: np.ndarray, prob=0.03):
    """椒盐噪声算法
    在图像里增加一些胡椒(pepper，黑块)和盐(salt, 白块)
    """
    noise = np.random.random(img.shape)
    salt = np.max(img)
    pepper = np.min(img)
    print(len(img[noise < prob/2]))
    img[noise < prob/2] = salt
    img[noise > 1-prob/2] = pepper
    return img


def wrap(img: np.ndarray, alpha=1):
    for i in range(28):
        for j in range(28):
            offset_x = int(alpha * norm[i]-0.5)
            offset_y = int(alpha * norm[i]-0.5)
            if i+offset_y < 28 and j+offset_x < 28:
                img[i, j] = img[(i+offset_y), (j+offset_x)]
    return img


# Standardize images across the dataset, mean=0, stdev=1
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# define data preparation
datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=False,
    # rotation_range=0,
    # width_shift_range=0.,
    # height_shift_range=0.,
    preprocessing_function=preprocess,
)
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
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
