from pickle import TRUE
import keras
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import random
from PIL import Image, ImageEnhance


norm = [-0.41, - 0.40, - 0.38, -0.37, -0.35, -0.33, -0.3, -0.28, -0.25,
        -0.23, -0.2, -0.17, -0.13, -0.1, -0.07, -0.03,
        0.0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2,
        0.23, 0.25, 0.28, 0.3, 0.33, 0.35, 0.37, 0.38, 0.39]


def noise(img: np.ndarray, sigma=0.05, mean=0) -> np.ndarray:
    x, y, z = img.shape
    mtx = sigma*np.random.randn(x, y, z)+mean
    img = img+mtx
    img = np.clip(img, 0, 1)
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    @param img 一个秩为3个Numpy张量，输出同尺寸
    """
    img = gaussian_filter(img, sigma=0.6)
    # img = noise(img)
    return img


datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # preprocessing_function=preprocess
)
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

datagen.fit(x_train)

label_dict = {0: 'apples', 1: 'aquarium fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottles', 10: 'bowls', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'cans', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'computer keyboard', 26: 'couch', 27: 'crab', 28: 'crocodile', 29: 'cups', 30: 'dinosaur', 31: 'dolphin', 32: 'elephant', 33: 'flatfish', 34: 'forest', 35: 'fox', 36: 'girl', 37: 'hamster', 38: 'house', 39: 'kangaroo', 40: 'lamp', 41: 'lawn-mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple', 48: 'motorcycle', 49: 'mountain',
              50: 'mouse', 51: 'mushrooms', 52: 'oak', 53: 'oranges', 54: 'orchids', 55: 'otter', 56: 'palm', 57: 'pears', 58: 'pickup truck', 59: 'pine', 60: 'plain', 61: 'plates', 62: 'poppies', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 70: 'roses', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflowers', 83: 'sweet peppers', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulips', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow', 97: 'wolf', 98: 'woman', 99: 'worm'}


def show_img(images, labels):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    idx = 0
    for t in range(0, images.shape[0], 16):
        for i, idx in enumerate(range(t, t+16)):
            ax = plt.subplot(4, 4, i+1)
            ax.imshow(images[idx], cmap='binary')
            title = label_dict[labels[idx][0]]  # 显示数字对应的类别
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        plt.show()


def gen_test():
    iter = datagen.flow(x_test, y_test, batch_size=10000)
    gen_test_X, gen_test_y = iter.next()
    return gen_test_X, gen_test_y


def test(X_test, Y_test):
    # validation_image_generator = datagen.flow(X_train, y_train, batch_size=100)
    Models = "/home/nil/Documents/code/AutomatedTesting2020/Models/CIFAR100/"
    test_RES = []
    for model_name in listdir(Models):
        print(model_name)
        model_path = Models+model_name
        model = keras.models.load_model(model_path)
        score = None
        model.compile(
            optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        score = model.evaluate(X_test, Y_test, verbose=1)
        test_RES.append([model_name, str(round(score[1], 4))])
    print('\n'.join(sorted([' '.join(s) for s in test_RES])))


def generate_img():
    for i, batch in enumerate(datagen.flow(x_test, y_test, batch_size=1,
                                           save_to_dir="Data/CIFAR100/Geometric",
                                           save_prefix="cifar", save_format='png')):
        i += 1
        if i > 50:
            break


if __name__ == "__main__":
    # gen_test_x, gen_test_y = gen_test()
    # show_img(x_train, y_train)
    # show_img(gen_test_x, gen_test_y)
    # test(gen_test_x, gen_test_y)
    generate_img()
