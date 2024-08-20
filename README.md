# Wine Quality Prediction using Multilayer Perceptron (MLP)

## Overview

This project aims to categorize wines based on their physicochemical attributes using a Multilayer Perceptron (MLP) implemented from scratch with NumPy. The Wine Quality Dataset, which includes various characteristics of wine samples and their quality ratings, is used for this purpose.

## Table of Contents

1. [Project Description](#project-description)
2. [Dataset Description](#dataset-description)
3. [Tasks](#tasks)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Project Description

In this project, we implement a basic MLP to predict the quality of red wine based on its physicochemical properties. The project involves data loading, preprocessing, model implementation, training, evaluation, and analysis.

## Dataset Description

The Wine Quality Dataset contains two sets of data: one for red wine and another for white wine. Each dataset includes several physicochemical attributes of wine samples, such as acidity, pH, and alcohol content, along with a quality rating. This project focuses on the red wine dataset.

Attributes:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (score between 0 and 10)

## Tasks

1. **Data Loading:** Load the red wine dataset using NumPy.
2. **Data Preprocessing:** Perform necessary preprocessing steps, such as normalization or standardization, and split the dataset into training and testing sets.
3. **Model Implementation:** Implement a Multilayer Perceptron (MLP) from scratch using only NumPy.
4. **Training:** Train the MLP model using the training data and implement the backpropagation algorithm.
5. **Evaluation:** Evaluate the model's performance using the testing data and calculate relevant metrics.
6. **Analysis:** Analyze the results and discuss observations or insights.
7. **Activation Functions:** Choose 10 different types of activation functions, plot their distributions, and their derivatives.

## Installation

To install the necessary packages, run the following commands:

```bash
pip install numpy pandas matplotlib seaborn
```

## Usage

To run the project, follow these steps:

1. Load the dataset and perform data preprocessing.
2. Implement and train the MLP model.
3. Evaluate the model's performance and analyze the results.
4. Plot the activation functions and their derivatives.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
