# Image Classification Project

## Overview
This project involves implementing and comparing different image classification models using the Fashion MNIST dataset. The project is divided into three main parts:
1. Convolutional Neural Networks (CNN)
2. Multi-Layer Perceptrons (MLP)
3. Exploratory Data Analysis (EDA)

## Dataset
The dataset used is the Fashion MNIST dataset, which contains 28x28 grayscale images of 10 different fashion items. Each image has a single label from the following categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

The dataset has 60,000 training images and 10,000 test images.

## 1. Convolutional Neural Networks (CNN)

### Steps
1. **Data Loading**
   - Import necessary libraries for deep learning (e.g., PyTorch).
   - Load the Fashion MNIST dataset.

2. **Data Preprocessing**
   - Normalize pixel values to a range between 0 and 1.
   - Split the dataset into training (80%), testing (10%), and validation (10%) sets.

3. **Model Architecture**
   - Design a simple CNN with convolutional layers, pooling layers, and fully connected layers.
   - Specify the input shape and the number of output classes.

4. **Compile the Model**
   - Choose an appropriate loss function for classification.
   - Select an optimizer and a metric for model evaluation.

5. **Model Training**
   - Train the CNN on the training dataset.
   - Monitor and adjust the training process as needed.

6. **Model Evaluation**
   - Evaluate the trained model on the testing dataset.
   - Calculate accuracy and other relevant metrics.

7. **Prediction**
   - Use the trained CNN to predict labels for a few test images.

8. **Discussion**
   - Discuss model performance and any challenges faced during training.
   - Mention potential improvements.

## 2. Multi-Layer Perceptrons (MLP)

### Steps
1. **Data Loading**
   - Import necessary libraries for deep learning.
   - Load the Fashion MNIST dataset.

2. **Data Preprocessing**
   - Flatten the 2D images into 1D arrays.
   - Normalize pixel values to a range between 0 and 1.
   - Split the dataset into training (80%), testing (10%), and validation (10%) sets.

3. **Model Architecture**
   - Design an MLP with multiple hidden layers.
   - Experiment with different activation functions and number of neurons.
   - Specify the input shape and the number of output classes.

4. **Compile the Model**
   - Select a suitable loss function for classification.
   - Choose an optimizer and a metric for model evaluation.

5. **Model Training**
   - Train the MLP on the training dataset.
   - Monitor and adjust hyperparameters as needed.

6. **Model Evaluation**
   - Evaluate the trained MLP on the testing dataset.
   - Calculate accuracy and other relevant metrics.

7. **Prediction**
   - Use the trained MLP to predict labels for a few test images.

8. **Discussion**
   - Compare MLP performance with the previously implemented CNN.
   - Discuss challenges faced and potential improvements.

## 3. Exploratory Data Analysis (EDA)

### Steps
1. **Data Distribution**
   - Visualize the distribution of classes in the dataset.
   - Examine class imbalance, if any.

2. **Image Statistics**
   - Calculate and visualize mean and standard deviation of pixel values.
   - Analyze brightness and contrast differences among classes.

3. **Dimensionality Analysis**
   - Explore image dimensions and resizing or cropping needs.
   - Visualize sample images to understand variability.

4. **Correlation Analysis**
   - Investigate correlations between pixel values within images.
   - Create a heatmap to visualize correlations.

5. **Noise and Artifacts**
   - Check for noise or artifacts in images.
   - Identify and visualize anomalies.

6. **Feature Engineering Possibilities**
   - Explore additional features that can be extracted from images.
   - Consider edge detection or texture analysis techniques.

7. **Data Augmentation**
   - Explore data augmentation techniques to increase training set diversity.
   - Visualize augmented images.

8. **Discussion**
   - Discuss how EDA findings may impact model design.
   - Consider preprocessing steps based on EDA results.

## Requirements
- Python 3.x
- Libraries: PyTorch, pandas, numpy, matplotlib, seaborn, wordcloud

## How to Run
1. Clone the repository.
2. Install the required libraries.
3. Run the Jupyter notebooks or Python scripts provided for each part of the project.

## Results
- Classification accuracy and other metrics for CNN and MLP models.
- Insights from EDA and its impact on model performance.
- Comparison of CNN and MLP models.

## Conclusion
This project demonstrates the implementation of CNN and MLP models for image classification and provides insights from exploratory data analysis to improve model performance.

