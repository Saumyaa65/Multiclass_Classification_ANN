import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score


#  Importing dataset
from tensorflow.keras.datasets import fashion_mnist

#  Loading dataset
(x_train, y_train), (x_test, y_test)= fashion_mnist.load_data()

#  x_train.shape= (60000,28,28), x_test.shape= (10000,28,28)
#  y_train.shape= (60000, ), y_test.shape= (10000, )

class_names=["0 Top/T-shirt", "1 Trouser", "2 Pullover", "3 Dress", "4 Coat",
             "5 Sandal", "6 Shirt", "7 Sneaker", "8 Bag", "9 Ankle boot"]

#  Data exploration
plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
# to open the image window: plt.show()

# Normalising dataset
x_train=x_train/255.0
x_test=x_test/255.0

plt.figure()
plt.imshow(x_train[1])
plt.colorbar()
# plt.show()

# Flattening dataset to convert into vector form, fewer dimensions
x_train=x_train.reshape(-1, 28*28)
x_test=x_test.reshape(-1, 28*28)
# x_train.shape= (60000, 784), x_test.shape= (10000, 784)

# Building the model

# Define object with Sequence of layers
model=tf.keras.models.Sequential()

# Adding first fully connected hidden layer:
# 1) units (no. of neurons)= 128 (determined experimentally)
# 2) activation function = ReLU (mostly in hidden layers)
# 3) input shape= 784 (flattened dataset)
model.add(tf.keras.layers.Dense(units=128,
                                activation='relu', input_shape=(784,)))

# Add second layer with dropout(regularization technique, prevents overfitting)
model.add(tf.keras.layers.Dropout(0.3)) # 30% random data discarded

# Adding output layer: 1) units= 10 (number of categories)
# 2) activation = softmax (multiple category classification)
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Training model

# Compiling model: 1) optimizer= adam, (minimize loss function)
# 2) loss function=sparse_categorical_crossentropy (acts as guide to optimizer)
# 3) metrics= sparse_categorical_accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
# To show details: print(model.summary())

# Training model, epochs= number of times it will train
model.fit(x_train, y_train, epochs=10)

# Model evaluation and prediction
test_loss, test_accuracy= model.evaluate(x_test,y_test)
print(f"Test Accuracy: {test_accuracy}")

# Model prediction
y_pred=np.argmax(model.predict(x_test),axis=-1)  # for multi classes
# y_pred = (model.predict(x_test) > 0.5).astype (‘int32’) (for binary classes)
print("Predicted and actual output examples:")
print(y_pred[-1], y_pred[4], y_pred[68], y_pred[467], y_pred[890])
print(y_test[-1], y_test[4], y_test[68], y_test[467], y_test[890])

# Confusion metrix
cm=confusion_matrix(y_test, y_pred)
print(cm)
acc_cm=accuracy_score(y_test, y_pred)
print(acc_cm)