# Semi-supervised Image Classification and Generation with PCA and K-Means
Micah Reich and David Krajewski (mreich@andrew.cmu.edu, dkrajews@andrew.cmu.edu)

Final project for 21-241 (Fall 2021)

## Abstract
Image classification and image generation are long-standing problems in the fields of artificial intelligence and computer science research. While many modern approaches to these tasks make use of deep neural networks like Generative Adversarial Networks and Convolutional Neural Networks, we demonstrate that simple image recognition and generation may be accomplished with classical methods in linear algebra, including Principal Component Analysis and K-means clustering.

## Introduction
Many modern approaches to image classification and image generation require deep neural networks that must be  trained iteratively, often consuming compute power and requiring large amounts of data. For our final project, we wish to demonstrate the power of unsupervised methods in classical machine learning, using PCA for low-rank approximation and embedding-space projections as well as a k-means clustering model for digit classification. Our method is able to correctly classify digits with high (85% to 90%) accuracy, noting the visual similarities between digits; we are also able to take advantage of the low-dimension embedding-space, using the k-means cluster centers and Gaussian noise to create novel MNIST image samples.

## Final Paper
To read the full paper, please visit [this link](https://github.com/micahreich/mnist-pca/blob/main/final-paper-v6.pdf).
