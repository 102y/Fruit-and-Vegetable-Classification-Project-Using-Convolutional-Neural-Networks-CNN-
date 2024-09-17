# Fruit-and-Vegetable-Classification-Project-Using-Convolutional-Neural-Networks-CNN-
The Fruit and Vegetable Classification Project using Convolutional Neural Networks (CNN) is based on two popular models VGG16 and VGG19 in the TensorFlow framework. The goal of the project is to build an AI system capable of classifying images containing different fruits and vegetables.
Project Description:
Purpose: Classify images of fruits and vegetables based on their visual features using pre-trained convolutional neural network models.
Models used:
VGG16: Convolutional neural network with 16 layers.
VGG19: Convolutional neural network with 19 layers.
Data:
The dataset contains pre-classified images of fruits and vegetables, divided into training, testing, and validation sets. Images of two different sizes are used:
Small size (100x100 pixels)
Original size (224x224 pixels)
Basic steps:
Data preparation:
Apply optimizations such as rotation, zooming, horizontal and vertical shifting to expand the dataset and make the model more accurate.
Model building:
Using pre-trained models (VGG16 and VGG19) with the base layers frozen to avoid changing their weights during training.
Adding new layers to analyze the characteristics of fruits and vegetables.
Model Training:
The model is trained using data of fruits and vegetables, where early stopping technique is used to stop training when there is a slight improvement in performance.
Model Testing:
The performance of the model is evaluated using the test dataset to determine its accuracy in classifying images.
Result Display:
Display the accuracy of the model in the training and testing stages using graphs.
Additional Features:
The ability to use external images to classify them based on the trained model.
Display predictive images with the predicted category (type of fruit or vegetable).
Benefits:
The project applications include improvements in the fields of agriculture, retail, and restaurants to improve the automatic classification of fruits and vegetables based on images.
