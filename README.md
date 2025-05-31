# Face Mask Detection using Deep Learning

## Project Overview

This project is a deep learning-based image classification system designed to detect whether individuals in images are wearing face masks. With the increasing need for automated public health tools, this model provides a quick and efficient way to verify mask compliance using computer vision.

## Objective

The main objective is to classify face images into two categories:
- **With Mask**
- **Without Mask**

By leveraging the power of transfer learning through a pre-trained **ResNet152V2** model, we achieve high accuracy in detecting face mask usage.

## Dataset

The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset), containing thousands of labeled images of people with and without masks. The dataset was:
- Downloaded and extracted within a Colab environment.
- Balanced across both classes to ensure fairness.
- Preprocessed to standardize the image size and format for model training.

## Model Architecture

The model is built using **TensorFlow** and **Keras**, with the following features:
- **ResNet152V2** used as the base model for feature extraction.
- Several dense layers with **ReLU** activation, **Batch Normalization**, and **Dropout** for regularization.
- Final layer with a **sigmoid activation** to output binary class predictions.

## Training and Evaluation

The dataset was split into training and test sets with an 80-20 ratio. The training set was further split for validation. The model was compiled using:
- **Adamax optimizer**
- **Sparse categorical crossentropy** loss function
- **Accuracy** as the evaluation metric

After training for 10 epochs, the model achieved:
- **Validation Accuracy**: Over 99%
- **Test Accuracy**: ~98.2%

## Prediction

The model includes functionality to predict mask usage on new input images. Users can upload an image, and the model will display it and print a prediction indicating whether a mask is present or not.

## Results Visualization

Training and validation accuracy and loss were plotted after training to help visualize model performance and spot any signs of overfitting.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, OpenCV, PIL
- Google Colab (runtime environment)
- Kaggle Datasets

## Conclusion

This project demonstrates the power of transfer learning in solving real-world problems such as face mask detection. It offers a practical example of how deep learning can be applied in the field of computer vision to aid in health and safety monitoring.
