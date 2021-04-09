#reference: https://www.tensorflow.org/tutorials/images/cnn
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#load and split dataset
(train_images, train_labels),(test_images, test_labels) = datasets.cifar10.load_data()

#normalize pixel values to be between 0 and 1
train_images, test_images = train_images /255.0, test_images/255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# IMG_INDEX = 7

# plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()

model = models.Sequential()

#CONVOLUTIONAL BASE
#input shape will be (32,32,3)
#32 filters of (3,3) over the input data
#relu activation function
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#perform max pooling operation using 2x2 samples and a stride of 2
model.add(layers.MaxPooling2D((2,2)))
#similar layers as the previous except its input is the output of the previous layer
#up the filters to 64 since the data shrinks and have more computtional depth
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))

# model.summary()

#Add dense layers
model.add(layers.Flatten())
#64 node dense layer
model.add(layers.Dense(64, activation='relu'))
#10 output layer for classes
model.add(layers.Dense(10))

# model.summary()


#train model:
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4,validation_data=(test_images, test_labels))



