# Multiclass Classification using Artificial Neural Network (ANN)

This project demonstrates **multiclass image classification** using a simple yet effective **Artificial Neural Network (ANN)**. The model is trained to classify images from the **Fashion MNIST dataset**, which consists of various articles of clothing.

## Overview

The goal is to accurately categorize images of apparel (like T-shirts, trousers, sneakers, etc.) into 10 distinct classes. This project provides a clear, hands-on example of how to preprocess image data, build a foundational neural network architecture for classification, and evaluate its performance on a common benchmark dataset.

## Features

* **Multiclass Image Classification:** Capable of classifying images into 10 different categories of clothing.
* **Artificial Neural Network (ANN):** Implements a fully connected neural network architecture for image classification.
* **Data Normalization:** Preprocesses image pixel data by scaling values to a common range (0-1), which improves model training stability and performance.
* **Data Flattening:** Transforms 2D image data into a 1D vector format suitable for input into a dense ANN.
* **Dropout Regularization:** Includes a dropout layer to prevent overfitting by randomly dropping a fraction of neuron connections during training.
* **Softmax Output Layer:** Uses a `softmax` activation function in the output layer to provide class probabilities for multiclass classification.
* **Performance Evaluation:** Assesses the model's accuracy using a confusion matrix and overall accuracy score.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib (for data visualization)
* Scikit-learn (`confusion_matrix`, `accuracy_score`)

## How It Works

The project follows these main steps:

1.  **Data Loading and Initial Exploration:**
    * The **Fashion MNIST dataset** is loaded directly using `tensorflow.keras.datasets`. This dataset comprises 60,000 training images and 10,000 test images, each 28x28 pixels, representing 10 types of clothing.
    * The `class_names` are defined for better readability of classification results.
    * Initial images can be displayed using `matplotlib` to understand the dataset (though `plt.show()` calls are commented out in the provided code).

2.  **Data Normalization and Flattening:**
    * Pixel values of the images (ranging from 0 to 255) are normalized by dividing by 255.0, scaling them to a range between 0 and 1. This is a crucial preprocessing step for neural networks.
    * The 2D image arrays (28x28) are `Flattened` into 1D vectors (784 features) to serve as input for the dense layers of the ANN.

3.  **Artificial Neural Network Construction:**
    * A Sequential Keras model is initialized.
    * **Input Layer and Hidden Layer:** The first layer is a `Dense` layer with 128 neurons and `relu` activation, taking the flattened image vector (784 inputs) as input.
    * **Dropout Layer:** A `Dropout` layer is added (with a 30% dropout rate) to randomly set a fraction of inputs to zero, helping to prevent overfitting.
    * **Output Layer:** The final layer is a `Dense` layer with 10 units (one for each class) and a `softmax` activation function. `softmax` outputs a probability distribution over the 10 classes, indicating the model's confidence for each category.

4.  **Model Compilation and Training:**
    * The model is compiled with the `adam` optimizer, `sparse_categorical_crossentropy` as the loss function (suitable for integer-labeled multiclass classification), and `sparse_categorical_accuracy` as the metric to monitor during training.
    * The model is then trained on the `x_train` and `y_train` data for 10 `epochs`.

5.  **Model Evaluation and Prediction:**
    * The trained model's performance is evaluated on the `x_test` dataset to determine its `test_loss` and `test_accuracy`.
    * Predictions are made on the test set, and `np.argmax` is used to convert the probability outputs into predicted class labels.
    * Sample predicted vs. actual outputs are printed for quick inspection.

6.  **Confusion Matrix:**
    * A **confusion matrix** is generated using `sklearn.metrics.confusion_matrix` to provide a detailed breakdown of correct and incorrect classifications for each digit class.
    * The **overall accuracy score** on the test set is also calculated.
