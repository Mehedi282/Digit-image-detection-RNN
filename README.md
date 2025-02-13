# MNIST Image Classification using RNN

## Overview

This project implements a **Recurrent Neural Network (RNN)** using **TensorFlow** and **Keras** to classify images from the **MNIST** dataset. The model uses RNN layers for image classification, showcasing an alternative to traditional CNN-based models. The model is trained on the MNIST dataset and can predict the digit in an uploaded image.

## Features

- Trains an RNN model on the MNIST dataset.
- Saves and loads the trained model (`mnist_rnn_model.h5`).
- Allows users to upload an image for prediction.
- Preprocesses the image before prediction.
- Displays the predicted class with **matplotlib**.

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **NumPy**
- **Matplotlib**
- **OpenCV** (for image preprocessing)
- **Google Colab** (for training and testing)

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- OpenCV

### Training the Model

To train the model on the MNIST dataset, run:

```bash
python train.py
