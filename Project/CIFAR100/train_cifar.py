from re import VERBOSE
import keras
import os
dataset = keras.datasets.cifar100.load_data()
Models = "/home/nil/Documents/code/AutomatedTesting2020/Models/CIFAR100/"
# model_name = "random2_cifar100.h5"

for model_name in os.listdir(Models):
    model_path = Models+model_name
    model = keras.models.load_model(model_path)
    x_train = dataset[0][0]
    y_train = dataset[0][1]
    x_test = dataset[1][0]
    y_test = dataset[1][1]
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=50, batch_size=64)
    if 'ResNet' in model_name:
        score = model.evaluate(x=x_test/255., y=y_test, verbose=1)
    else:
        score = model.evaluate(x=x_test, y=y_test, verbose=1)
    print(model_name)
    print(score)
