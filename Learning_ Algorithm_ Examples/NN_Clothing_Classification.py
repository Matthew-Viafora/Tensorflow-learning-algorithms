#Based on Tensorflow documentation for "Basic classification: Classify images of clothing": 
#https://www.tensorflow.org/tutorials/keras/classification
#Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

#helper libraries
import numpy as np
import matplotlib.pyplot as plt

#load dataset
fashion_mnist = keras.datasets.fashion_mnist

#split into testing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(train_images.shape)
# print(train_images[0,23,23])
# print(train_labels[:10])

#labels
class_names = ['T-shrit/top', 'Trouser', 'pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#show some training images:
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

#preprocess our data (Normalize between 0 and 1):
train_images = train_images / 255.0
test_images = test_images / 255.0


#build model with keras
#three layers (input, hidden, output)
#input shape is (28,28), but flattened to a vector of 784 for input
#relu activation function for hidden layer
#softmax activation function for output to obtain probabilities
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #input layer (1)
    keras.layers.Dense(128, activation='relu'), #hidden layer (2)
    keras.layers.Dense(10, activation='softmax') #output layer (3)
])

#compile model with adam optimizer and 'sparse categorical crossentropy cost function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#train the model! (so cool!!!) pass training images, labels, and epochs
model.fit(train_images, train_labels, epochs=10)

#evaluation of the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print('Test accuracy:', test_acc)

#making predictions
predictions = model.predict(test_images)
