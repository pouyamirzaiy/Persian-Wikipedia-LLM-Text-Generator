# Ad Click Prediction Project

## Overview

This project aims to predict the likelihood of a user clicking on an online advertisement. The dataset contains 10 columns, with 9 of them representing an instance of a data record as features. The target variable is the "Clicked on Ad" column, which indicates whether the visitor clicked on the ad.

## Table of Contents

1. [Project Description](#project-description)
2. [Feature Engineering Steps](#feature-engineering-steps)
3. [Model Evaluation Results](#model-evaluation-results)
4. [Analysis](#analysis)
5. [Task](#task)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)

## Project Description

Click-Through Rate (CTR) prediction is a critical aspect of online advertising, directly impacting platform revenue and the success of marketing campaigns. In this project, we explore user interaction data to predict the likelihood of a user clicking on an online advertisement.

## Feature Engineering Steps

1. **Extracting Hour, Day, and Month from Timestamp:**

   - Converted the 'Timestamp' column to datetime format.
   - Extracted the hour, day, and month components.
   - These features allow us to account for time-related patterns in ad interactions.

2. **Handling Categorical Variables:**

   - Addressed categorical columns ('Gender', 'Country', 'City', 'Ad Topic Line').
   - Imputed missing values with 'missing' and applied one-hot encoding.
   - Helps incorporate categorical information into the model.

3. **Cyclical Encoding for Time Columns:**

   - Transformed time-related columns ('Hour', 'Day', 'Month') using cyclical features.
   - Captures cyclic patterns (e.g., daily or monthly) without imposing linearity.

4. **Imputing and Scaling Numerical Features:**
   - Handled numerical features ('Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage').
   - Imputed missing values with the mean.
   - Scaled features to a common range (MinMax scaling).

## Model Evaluation Results

- **Random Forest Classifier:**

  - Trained the model using the preprocessed features.
  - Achieved an accuracy of approximately 87.47% on the test set.

- **XGBoost Model:**

  - Best Parameters: {'classifier**colsample_bytree': 0.6, 'classifier**learning_rate': 0.1, 'classifier**max_depth': 7, 'classifier**n_estimators': 300, 'classifier\_\_subsample': 0.8}
  - Accuracy: 87.5%
  - Confusion Matrix:
    ```
    [[1333  185]
     [ 182 1236]]
    ```
  - Classification Report:

    ```
                  precision    recall  f1-score   support

             0       0.88      0.88      0.88      1518
             1       0.87      0.87      0.87      1418

      accuracy                           0.88      2936
     macro avg       0.87      0.87      0.87      2936
    weighted avg       0.88      0.88      0.88      2936
    ```

## Analysis

- The Random Forest and XGBoost models perform well, achieving balanced precision and recall for both classes.
- The cyclical encoding of time features allows us to capture cyclic patterns without introducing linearity assumptions.
- Further exploration could involve hyperparameter tuning and experimenting with other classifiers.

## Task

CTR prediction is a critical aspect of online advertising, directly impacting platform revenue and the success of marketing campaigns. In this task, you will explore the intricacies of user interaction data to predict the likelihood of a user clicking on an online advertisement.

### Steps:

1. **Exploratory Data Analysis (EDA):**

   - Conduct a comprehensive report documenting your EDA findings and insights (in a separate document with visualizations).

2. **Data Cleaning and Feature Engineering:**

   - Perform data cleaning and feature engineering, documenting your steps thoroughly and justifying your decisions (in your Jupyter notebook).

3. **Classification Algorithm Implementation:**

   - Implement a classification algorithm to predict click-through events and evaluate model performance using appropriate metrics.
   - Experiment with different algorithms and compare their performance.

4. **Cross-Validation:**
   - Use cross-validation techniques to evaluate your model and report your results.

## Installation

To install the necessary packages, run the following commands:

```bash
pip install pandas numpy matplotlib seaborn skimpy scipy statsmodels scikit-learn feature_engine xgboost catboost lightgbm
```

## Usage

To run the project, follow these steps:

1. Load the dataset and perform EDA.
2. Clean the data and engineer features.
3. Train and evaluate the classification models.
4. Document your findings and results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## **Contact Information**

If you have any questions or feedback, feel free to reach out to me at pouya.8226@gmail.com
