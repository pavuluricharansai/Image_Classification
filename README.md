# CIFAR-10 Image Classification with CNN

This project demonstrates how to build a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 different classes. The model is trained and evaluated with metrics such as accuracy, loss, confusion matrix, and a classification report.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Training Process](#training-process)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

This project performs the following steps:

1. **Data Loading**: The CIFAR-10 dataset is loaded and normalized using the `torchvision` library.
2. **Model Definition**: A simple CNN model is defined with three convolutional layers followed by fully connected layers.
3. **Model Training**: The model is trained using the Adam optimizer and Cross-Entropy Loss. The training loop runs for 10 epochs.
4. **Evaluation**: After training, the model is evaluated using validation accuracy, confusion matrix, and classification report.
5. **Visualizations**: Loss and accuracy plots, as well as sample predictions and a confusion matrix, are generated to assess the model's performan
