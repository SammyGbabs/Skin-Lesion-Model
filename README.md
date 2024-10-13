# Skin-Lesion-Model

## Project Overview

**Objectives**

The primary objective of this project is to develop a robust image classification model that accurately distinguishes between four different skin conditions: **Monkeypox, Measles, Chickenpox, and Normal,** just by uploading a skin lesion. By leveraging a Convolutional Neural Network (CNN) architecture, the project aims to:

**1) Build a Deep Learning Model:** Create an effective image classification model utilizing CNN to identify and classify images into the specified categories.

**2) Optimize Model Performance:** Experiment with various optimization techniques (including Adam and RMSprop), regularization methods (L1 and L2 regularization, dropout), and hyperparameter tuning to improve accuracy and minimize loss.

**3) Evaluate Model Effectiveness**: Assess the model's performance using key metrics such as accuracy, precision, recall, and F1 score to ensure reliable classification capabilities.

**4) Perform Error Analysis:** Utilize confusion matrices to visualize and analyze classification errors, helping identify areas for improvement in model performance.

**5) Save and Deploy Model:** Ensure that the trained model can be easily saved and deployed for future use or further enhancements.

## Dataset Overview
The dataset used for this project is an image collection obtained from Kaggle, focusing on distinguishing between four specific skin conditions: Monkeypox, Measles, Chickenpox, and Normal. Each image in the dataset is pre-labeled, providing a solid foundation for training the classification model.

During the data processing phase, I took the initiative to split the dataset into training, validation, and test sets, as the original dataset from Kaggle was not pre-split. This division ensures a balanced and effective approach to training and evaluating the model. The training set comprises 80% of the total images, enabling the model to learn from a diverse range of examples. The validation set, making up 10% of the dataset, is used to fine-tune hyperparameters and assess the model's performance during training. The remaining 10% constitutes the test set, which is essential for a final evaluation of the model's accuracy and robustness.

By leveraging this structured approach to data splitting and utilizing a dataset rich in labeled images, the project aims to develop a reliable model for the accurate classification of these skin conditions.

Link to Dataset:  https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset

## Key Findings and Discussion on my Optimization technique and Parameters setting

The different combinations of optimization techniques and parameter settings:

1. **Combination 1**: Early stopping, RMSprop, L2 regularization (0.001)  
   - **Accuracy**: 65.62%  
   - **Loss**: 0.7824  

2. **Combination 2**: Early stopping, dropout (0.5), RMSprop, L1 regularization (0.001)  
   - **Accuracy**: 70.96%  
   - **Loss**: 0.8067  

3. **Combination 3**: Early stopping, dropout (0.5), RMSprop, L2 regularization (0.001)  
   - **Accuracy**: 65.62%  
   - **Loss**: 0.7824  

4. **Combination 4**: Early stopping, dropout (0.5), Adam, L2 regularization (0.001)  
   - **Accuracy**: 58.59%  
   - **Loss**: 0.9957  

5. **Combination 5**: Early stopping, dropout (0.5), Adam, L1 regularization (0.01)  
   - **Accuracy**: 56.12%  
   - **Loss**: 1.0843  

6. **Combination 6**: Early stopping, Adam, L1 regularization (0.01)  
   - **Accuracy**: 65.62%  
   - **Loss**: 1.1293  

7. **Combination 7**: Early stopping, Adam, L1 regularization (0.001)  
   - **Accuracy**: 60.81%  
   - **Loss**: 0.9270

In this project, several combinations of optimization techniques and parameter settings were tested to improve the performance of the model. Below is a detailed discussion of the underlying principles behind each optimization technique and the rationale for the parameter selections:

### 1. **Early Stopping**
   - **Principle**: Early stopping is a form of regularization used to prevent overfitting. It monitors the model's performance on a validation set and halts training when the performance ceases to improve after a certain number of epochs. This helps avoid overfitting and saves computational resources.
   - **Relevance**: This technique ensures that the model does not train too long, which could lead to overfitting, especially given the complexity of neural networks.
   - **Parameters**:
     - `monitor='val_loss'`: The model’s validation loss was monitored to decide when to stop training.
     - `patience`: This parameter specifies how many epochs the model will continue training after the performance stops improving.
   
   Early stopping was applied across all experiments to ensure that the model does not overfit to the training data and generalizes well to new, unseen data.

### 2. **Optimization Algorithms**
   - **RMSprop**: 
     - **Principle**: RMSprop (Root Mean Square Propagation) adjusts the learning rate for each parameter individually, using the square root of the average of squared gradients. It helps in dealing with the "vanishing gradient" problem by dividing the gradient by a moving average of its magnitude.
     - **Relevance**: RMSprop is well-suited for problems involving noisy or non-stationary objectives, making it a good fit for training deep networks where gradients can be unstable.
   - **Adam**: 
     - **Principle**: Adam (Adaptive Moment Estimation) is an adaptive learning rate optimizer that combines the benefits of RMSprop and momentum. It maintains running averages of both the gradient and its square, allowing for a faster and more reliable convergence.
     - **Relevance**: Adam is widely used in deep learning because of its efficiency and effectiveness across a variety of tasks.
   - **Comparison**: RMSprop performed better in most cases in this project, as seen in the second and third combinations (accuracy: 70.96% and 65.62%), while Adam did not perform as well (accuracy: 58.59% and lower).

### 3. **Regularization**
   - **L1 Regularization (Lasso)**: 
     - **Principle**: L1 regularization adds the absolute value of the magnitude of coefficients as a penalty term to the loss function. This forces some coefficients to be exactly zero, effectively performing feature selection.
     - **Relevance**: L1 regularization helps in sparse models and can eliminate irrelevant features or parameters, leading to simpler and less overfitted models.
   - **L2 Regularization (Ridge)**: 
     - **Principle**: L2 regularization adds the squared value of the coefficients as a penalty to the loss function. Unlike L1, it does not eliminate features but reduces the magnitude of parameters, leading to more generalized models.
     - **Relevance**: L2 regularization helps prevent overfitting by shrinking large weights, thereby making the model more robust to noise and improving generalization.
   - **Comparison**:
     - The second combination using **L1 regularization (0.001)** achieved an accuracy of 70.96% with a loss of 0.8067. This indicates that L1 regularization was effective in penalizing large weights and preventing overfitting.
     - **L2 regularization (0.001)**, as seen in the third and first combinations, performed decently but did not outperform L1 regularization.

### 4. **Dropout**
   - **Principle**: Dropout is a regularization technique that randomly drops units (along with their connections) from the neural network during training. It prevents co-adaptation of hidden units by forcing the network to learn more robust features.
   - **Relevance**: Dropout was used to combat overfitting, particularly in models with a large number of parameters. It forces the network to learn multiple independent representations of the data.
   - **Dropout Rate**: A dropout rate of 0.5 was used, which is a commonly used value to balance regularization while maintaining network capacity. The second combination showed that dropout (0.5) paired with RMSprop and L1 regularization led to the best performance.

### 5. **Parameter Selection and Tuning**
   - **L1 and L2 Regularization Values**: 
     - L1 regularization was set to `0.001` in some experiments and `0.01` in others. The smaller value (0.001) performed better in terms of accuracy (70.96%), while the larger value (0.01) resulted in poorer performance, as seen in the fifth and sixth combinations.
     - L2 regularization also used a value of `0.001`. It performed decently (65.62% accuracy), but L1 regularization was more effective in this project.
   - **Optimizer Selection**: RMSprop consistently outperformed Adam, with the second combination (RMSprop + L1) achieving the highest accuracy. This indicates that RMSprop’s adaptation to learning rates during training was more suitable for the dataset and model architecture.

### Summary of Results:
- **Best Model**: The **second combination** (early stopping, dropout of 0.5, RMSprop, L1 regularization of 0.001) achieved the best performance, with an accuracy of **70.96%** and a loss of **0.8067**.
- **Optimization Insights**:
  - **RMSprop** outperformed **Adam** in most cases, especially when paired with L1 regularization and dropout.
  - **L1 regularization (0.001)** provided the best performance in controlling overfitting while maintaining good accuracy.
  - **Dropout** at 0.5 was an effective regularization technique across various configurations.

This thorough analysis shows that combining **RMSprop** with **L1 regularization** and **dropout** produced the most robust model in terms of both accuracy and loss.

Thank you for sharing the confusion matrices for both models. Below is a comprehensive error analysis based on the metrics and confusion matrices you've provided:

## Error Analysis

#### 1. **Confusion Matrices Overview**
- **Model with Optimization Technique**:
  - The confusion matrix indicates that the model performs better across most classes. It has correctly classified a significant number of instances from Class 2 (measles) and Class 3 (normal). However, there are misclassifications, especially for Class 0 (monkeypox) and Class 1 (chickenpox).
  - Specifically, Class 0 has a substantial number of misclassifications into Class 2, suggesting confusion between monkeypox and measles.
  - Class 1 has only a few instances correctly identified, indicating difficulty in distinguishing chickenpox from the other classes.

- **Vanilla Model**:
  - The performance is noticeably lower across all classes. The model has a high number of misclassifications, particularly for Class 3, where several instances are confused with Class 0 and Class 1.
  - Class 2 also suffers from misclassifications, indicating that the vanilla model struggles to learn distinguishing features of the different classes.

#### 2. **Performance Metrics**
- **Vanilla Model**:
  - **Accuracy**: 35.42%
  - **Precision (Macro)**: 14.17%
  - **Recall (Macro)**: 23.59%
  - **F1 Score (Macro)**: 14.29%

  The low metrics indicate that the vanilla model struggles significantly to classify the images correctly. The precision and recall suggest that the model is particularly poor at identifying positive instances correctly.

- **Model with Optimization Technique**:
  - **Accuracy**: 67.71%
  - **Precision (Macro)**: 65.89%
  - **Recall (Macro)**: 50.18%
  - **F1 Score (Macro)**: 48.22%

  The improved performance metrics for the optimized model reflect successful adjustments in training, leading to better class distinction. The precision score indicates that the model is reasonably accurate when it predicts a class, while the recall shows room for improvement in capturing all relevant instances.

Here’s the data presented in a tabular format:

| Model                         | Accuracy  | Precision (Macro) | Recall (Macro) | F1 Score (Macro) |
|-------------------------------|-----------|-------------------|----------------|-------------------|
| Vanilla Model                 | 35.42%    | 14.17%            | 23.59%         | 14.29%            |
| Model with Optimization Technique | 67.71% | 65.89%            | 50.18%         | 48.22%            |

#### 3. **Specificity and Additional Metrics**
Calculating specificity for each class can provide a more comprehensive evaluation of the model's performance in terms of true negative rates. This analysis helps identify classes that are frequently confused with others, offering insights into areas where further improvements may be necessary..

### Summary
The implementation of optimization techniques has significantly enhanced model performance, as evidenced by the metrics and confusion matrix. The optimized model achieves higher accuracy and better precision and recall compared to the vanilla model. However, further tuning and additional techniques may still be necessary to reduce confusion among specific classes, particularly between monkeypox and measles, as well as chickenpox and the other classes.

The best model, which combined early stopping, dropout of 0.5, RMSprop, and L1 regularization of 0.001, achieved an accuracy of 70.96% and a loss of 0.8067. Model performance metrics highlighted significant improvements, with the vanilla model showing an accuracy of 0.3542 compared to 0.6771 after applying optimization techniques. Additional metrics, including precision, recall, and F1 score, further illustrated the enhancements achieved through optimization.

It is important to note that the dataset used in this project is relatively small, consisting of only 770 images. The limited size of the dataset likely constrained the model's performance, and an increase in the number of training images could significantly enhance the model's accuracy and generalization capabilities. Thus, dataset size plays a crucial role in the effectiveness of the model..

Here's a section to include in the README for running the notebook and loading the saved models:

---

### Instructions for Running the Notebook

1. **Environment Setup:**
   - Make sure all dependencies are installed by running the following:
     ```bash
     pip install -r requirements.txt
     ```
   
2. **Running the Notebook:**
   - Open the Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Navigate to the notebook file (`your_notebook_name.ipynb`) and open it.
   
   - Run the notebook cells sequentially to execute the code, train the models, and evaluate the results.

3. **Loading Pre-trained Models:**
   - Pre-trained models are saved in the directory `saved_models/`. To load the models and evaluate them without re-training:
     ```python
     from tensorflow.keras.models import load_model

     # Load the optimized model
     optimized_model = load_model('saved_models/optimized_model.h5')

     # Load the vanilla model
     vanilla_model = load_model('saved_models/vanilla_model.h5')
     ```

4. **Model Evaluation:**
   - After loading the model, use the following commands to evaluate it:
     ```python
     # Evaluate the model on the test set
     loss, accuracy = optimized_model.evaluate(X_test, y_test)
     print(f"Optimized Model - Loss: {loss}, Accuracy: {accuracy}")

     # Similarly for the vanilla model
     loss, accuracy = vanilla_model.evaluate(X_test, y_test)
     print(f"Vanilla Model - Loss: {loss}, Accuracy: {accuracy}")
     ```

---

This section should give users clear instructions on how to run the notebook and load the pre-trained models.
