# Skin-Lesion-Model

## Project Overview

**Objectives**

The primary objective of this project is to develop a robust image classification model that accurately distinguishes between four different skin conditions: Monkeypox, Measles, Chickenpox, and Normal, just by uploading a skin lesion. By leveraging a Convolutional Neural Network (CNN) architecture, the project aims to:

1) Build a Deep Learning Model: Create an effective image classification model utilizing CNN to identify and classify images into the specified categories.
2) Optimize Model Performance: Experiment with various optimization techniques (including Adam and RMSprop), regularization methods (L1 and L2 regularization, dropout), and hyperparameter tuning to improve accuracy and minimize loss.
3) Evaluate Model Effectiveness: Assess the model's performance using key metrics such as accuracy, precision, recall, and F1 score to ensure reliable classification capabilities.
4) Perform Error Analysis: Utilize confusion matrices to visualize and analyze classification errors, helping identify areas for improvement in model performance.
5) Save and Deploy Model: Ensure that the trained model can be easily saved and deployed for future use or further enhancements.

## Dataset Overview
The dataset used for this project is an image collection obtained from Kaggle, focusing on distinguishing between four specific skin conditions: Monkeypox, Measles, Chickenpox, and Normal. Each image in the dataset is pre-labeled, providing a solid foundation for training the classification model.

During the data processing phase, I took the initiative to split the dataset into training, validation, and test sets, as the original dataset from Kaggle was not pre-split. This division ensures a balanced and effective approach to training and evaluating the model. The training set comprises 80% of the total images, enabling the model to learn from a diverse range of examples. The validation set, making up 10% of the dataset, is used to fine-tune hyperparameters and assess the model's performance during training. The remaining 10% constitutes the test set, which is essential for a final evaluation of the model's accuracy and robustness.

By leveraging this structured approach to data splitting and utilizing a dataset rich in labeled images, the project aims to develop a reliable model for the accurate classification of these skin conditions.

Link to Dataset:  https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset
