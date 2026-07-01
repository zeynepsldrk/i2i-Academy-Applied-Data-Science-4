# Student Lifestyle and Academic Performance Classification

## Overview
The project focuses on predicting student academic success based on lifestyle metrics using scikit-learn.

## Dataset
* **Name:** Student Lifestyle and Academic Performance Dataset
* **Source:** Kaggle (`rafi003/student-lifestyle-and-academic-performance-dataset`)
* **Task:** Binary Classification. The continuous `CGPA` variable is thresholded at 7.0 to create a binary target (`Performance_Class`: 1 for High Performance, 0 for Low Performance).

## Stack
* Python 3
* pandas, numpy (Data manipulation)
* scikit-learn (Modeling and preprocessing)
* kagglehub (Data ingestion)

## Models
The pipeline trains and evaluates two distinct algorithms:
1. Logistic Regression
2. Random Forest Classifier

## Results
Models were evaluated on a 20% holdout test set using Accuracy and Confusion Matrix metrics.
* **Logistic Regression:** 82.00% Accuracy
* **Random Forest:** 78.50% Accuracy

Logistic Regression outperformed Random Forest on this dataset. The implementation of `StandardScaler` significantly improved the convergence and generalization of the distance-based Logistic Regression model. Conversely, the tree-based Random Forest model exhibited slight overfitting on the training data, leading to a minor drop in test accuracy.

## Installation and Usage

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn kagglehub
```

2. Execute the pipeline:
```bash
python classification_task.py
```
*Note: The script automatically fetches the required dataset via the Kaggle API before executing the training pipeline.*
