---
layout: post
title: Applying Deep Neural Network on the MNIST dataset
---

A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layer. Deep neural
networks have been used on a variety of tasks, including computer vision, speech recognition, natural language processing, machine 
translation, audio recognition, social network filtering, playing board and video games and medical diagnosis. Deep neural networks 
have produced results comparable to and in some cases superior to human experts. In this article we will apply deep neural network on 
MNIST dataset. MNIST is the most studied dataset (<a href='https://yann.lecun.com/exdb/mnist/' target="_blank">MNIST</a>) in computer 
vision tasks.

The state of the art result for MNIST dataset has an accuracy of 99.79%. In this article, we will achieve an accuracy of 98.71%.

## MNIST Dataset
![MNIST](https://raw.githubusercontent.com/ZainAmin/zainamin.github.io/master/images/mnistimage.png "MNIST")

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It has 60,000 grayscale images under the training set and 10,000 grayscale images under the test set. We will use the Keras library with Tensorflow backend to classify the images.

## Installing Keras and Tensorflow

Keras is an open source neural network library written in Python.It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, or Theano. Designed to enable fast experimentation with deep neural networks. To learn more about it, visit there official <a href="https://www.tensorflow.org/">website</a>.

Tensorflow was developed by the Google Brain team. To learn more about it, visit there official <a href="https://www.tensorflow.org/">website</a>.

# Implementation

First, we need to import all the libraries required.

```python
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
```

The MNIST dataset is provided by Keras.

```python
# 28x28 images of hand-written digits 0-9
mnist = tf.keras.datasets.mnist
# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
Actual data to pass through the deep neural network. Pixel values range from 0-255

```python
print(x_train[0])
```
![MNIST](https://raw.githubusercontent.com/ZainAmin/zainamin.github.io/master/images/imagepixeldata.PNG "MNIST")

Actual image of the Above Data

```python
plt.imshow(x_train[0])
plt.show()
```
![MNIST](https://raw.githubusercontent.com/ZainAmin/zainamin.github.io/master/images/actualimagemnist.PNG "MNIST")

Binary form of above image

```python
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
```
![MNIST](https://raw.githubusercontent.com/ZainAmin/zainamin.github.io/master/images/actualimagebinary.PNG "MNIST")

Now we have to normalize the image data to scale pixel values between 0-1 instead of 0-255

```python
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
```

Now the actual pixel values we have to pass through the deep neural network will be between 0-1 i.e. in binary form, as shown in figure below:





