import keras
dataset = keras.datasets.mnist.load_data()
# model_path = "/home/nil/Desktop/Model/MNIST/dnn_with_dropout.hdf5"
Models = "/home/nil/Documents/code/AutomatedTesting2020/Models"
model_path = Models+"/MNIST/dnn_with_dropout.hdf5"

model = keras.models.load_model(model_path)
x_train = dataset[0][0].reshape(60000, -1)
y_train = dataset[0][1]
x_test = dataset[1][0].reshape(10000, -1)
y_test = dataset[1][1]
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=10, batch_size=100)
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(score)
