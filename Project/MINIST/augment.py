import keras
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import numpy as np
import random
from os import listdir
from scipy.ndimage.filters import gaussian_filter


norm = [-0.38, -0.37, -0.35, -0.33, -0.3, -0.28, -0.25,
        -0.23, -0.2, -0.17, -0.13, -0.1, -0.07, -0.03,
        0.0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2,
        0.23, 0.25, 0.28, 0.3, 0.33, 0.35, 0.37]


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
        # break


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    @param img 一个秩为3个Numpy张量，输出同尺寸
    """
    # img = pepper_noise(img)
    img = wrap(img)
    # img = gaussian_filter(img, sigma=2)
    return img


def pepper_noise(img: np.ndarray, prob=0.02):
    """椒盐噪声算法
    在图像里增加一些胡椒(pepper，黑块)和盐(salt, 白块)
    @param prob: 在图像里添加椒盐噪声的百分比
    """
    noise = np.random.random(img.shape)
    img[noise < prob/2] = 254.0
    img[noise > 1-prob/2] = 0.0
    return img


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

# define data preparation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
    # rotation_range=15,
    # width_shift_range=0.15,
    # height_shift_range=0.15,
    preprocessing_function=preprocess,
    fill_mode='nearest',
)
# fit parameters from data
datagen.fit(X_test)
# show_img()

testgen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=False,
)


def generate_train():
    iter = datagen.flow(X_train, y_train, batch_size=20000)
    gen_train_X, gen_train_y = iter.next()
    Models = "/home/nil/Documents/code/AutomatedTesting2020/Models/MNIST/"
    for model_name in listdir(Models):
        print(model_name)
        model_path = Models+model_name
        model = keras.models.load_model(model_path)
        model.fit()
        score = None
        model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if(model_name.split('_')[0] in ["lenet5", "vgg16", "random1", "random2"]):
            score = model.evaluate(X_test, y_test, verbose=1)
        if (model_name.split('_')[0] in ["dnn"]):
            score = model.evaluate(X_test.reshape(-1, 784), y_test, verbose=0)


def generate_test():
    iter = datagen.flow(X_test, y_test, batch_size=10000)
    gen_test_X, gen_test_y = iter.next()
    # validation_image_generator = datagen.flow(X_train, y_train, batch_size=100)
    Models = "/home/nil/Documents/code/AutomatedTesting2020/Models/MNIST/"

    train_RES = []
    for model_name in listdir(Models):
        print(model_name)
        model_path = Models+model_name
        model = keras.models.load_model(model_path)
        score = None
        model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if(model_name.split('_')[0] in ["lenet5", "vgg16", "random1", "random2"]):
            score = model.evaluate(gen_test_X, gen_test_y, verbose=1)
        if (model_name.split('_')[0] in ["dnn"]):
            score = model.evaluate(gen_test_X.reshape(-1, 784), gen_test_y, verbose=0)
        train_RES.append([model_name, str(round(score[1], 4))])
    print('\n'.join(sorted([' '.join(s) for s in train_RES])))


generate_test()
