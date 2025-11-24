# Fraud-Analysis-and-Prediction-PyTorch

This repository contains an end-to-end solution for detecting fraudulent transactions in e-commerce. The workflow spans from interactive Exploratory Data Analysis (EDA) to the implementation of a Deep Learning model for binary classification.

##  Data Source

The dataset used for this project contains a history of e-commerce transactions.
* **Source:** [E-commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)
* **Problem:** Binary classification on an imbalanced dataset (identifying the minority class: Fraud).

##  Key Features
* Project Workflow
1. EDA (Exploratory Data Analysis)

Inspection of data types, missing values, and variable distributions.

Interactive visualization using Plotly to explore relationships between features and the target label.

Analysis of class imbalance (Fraud vs. Non-Fraud) and identification of outliers.

2. Preprocessing

Scaling of numerical variables (using Scikit-Learn).

Encoding of categorical variables (One-hot/Label encoding).

Train/test split with stratification to maintain class proportions.

Conversion of data into PyTorch Tensors.

3. Modeling with PyTorch

Custom Neural Network (ANN) with linear layers and activation functions (ReLU/Sigmoid).

Implementation of regularization techniques (Dropout/Batch Normalization).

Training loop optimization with loss tracking (Binary Cross Entropy).

Handling class imbalance (e.g., using class weights).

4. Model Evaluation

Classification report with a focus on Precision, Recall, and F1-score for the minority class.

High Overall Performance: All three trained models showed excellent performance, significantly exceeding the random baseline (AUC of 0.5, the dotted gray line). This indicates that they all successfully learned effective patterns for distinguishing fraud.

5. Model Comparison:

Random Forest achieved the highest score with an AUC of 0.9713, demonstrating the best ability to maximize true positives (detecting fraud) while minimizing false positives (avoiding disturbing legitimate customers) in the initial stages.

PyTorch FNN (Our Model): Achieved an AUC of 0.9628, an extremely competitive result and very close to that of Random Forest. This validates that the designed neural network architecture is robust and effective for this problem.

Logistic Regression: Served as a solid baseline with an AUC of 0.9397.


Confusion matrix to visualize False Negatives vs. False Positives.

Performance visualization (Loss curves over epochs).

How to Run
Clone this repository or download the notebook.

Download the dataset from Kaggle and place the file transactions.csv in the same directory as the notebook.

Install the required dependencies:

   ```bash
   pip install pandas numpy plotly matplotlib seaborn torch
