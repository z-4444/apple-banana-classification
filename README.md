# Apple vs. Banana Classification using MobileNet
## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Data Preprocessing](#data-preprocessing)
5. [Training](#training)
6. [Results](#results)
7. [Dependencies](#dependencies)
8. [Classification Report](#classification-report)

## Introduction
This is a deep learning model for classifying images of apples and bananas using the MobileNet architecture in TensorFlow. The model is trained on a dataset of images of apples and bananas, and it can be used to predict the class of new images.

## Dataset
The dataset consists of two classes: apples and bananas. It contains 1,000 images for each class, divided into training and test sets.

# Model Architecture
The model architecture consists of a modified MobileNet architecture, with a modified first convolutional layer and additional preprocessing layers. The model uses global average pooling instead of fully connected layers, and a sigmoid activation function in the output layer.

The MobileNet architecture is pre-trained on the ImageNet dataset, and the weights of the pre-trained model are used as initial values for the training of the apple vs. banana classification model. The weights of the last few layers of the pre-trained model are fine-tuned during the training of the classification model.

# Data Preprocessing
The images are preprocessed using the ` ImageDataGenerator ` class from TensorFlow. The ` ImageDataGenerator ` is used to rescale the pixel values of the images, as well as perform data augmentation techniques such as rotation, shifting, and flipping.

In addition to rescaling, the images are also preprocessed using a custom function that converts the RGB image to HSV and YCbCr color spaces, and concatenates them with the original RGB image. This is done to improve the robustness of the model to changes in lighting conditions.

# Training
The model is trained using the fit method of the Model class in TensorFlow. The training is done using batches of size 32, for a total of 1,000 epochs. The Adam optimizer is used, with a learning rate of 1e-4, and binary cross-entropy loss. The training progress is monitored using the validation set.

# Results
The model is evaluated on the test set using the evaluate method of the Model class in TensorFlow. The test accuracy is reported as a percentage.

# Dependencies
The following are the dependencies required for this project:

* TensorFlow
* NumPy
* Matplotlib

## Classification Report

```
Confusion Matrix:
 [[47  0]
 [ 0 44]]

Classification Report:

              precision    recall  f1-score   support

       apple       1.00      1.00      1.00        47
      banana       1.00      1.00      1.00        44

    accuracy                           1.00        91
   macro avg       1.00      1.00      1.00        91
weighted avg       1.00      1.00      1.00        91  ```
