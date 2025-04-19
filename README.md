# Titanic - Machine Learning from Disaster: Custom XGBoost Implementation

This repository contains a custom implementation of the XGBoost algorithm designed for the Kaggle "Titanic - Machine Learning from Disaster" competition. The goal of this project was to predict passenger survival on the Titanic using machine learning techniques, while also gaining a deeper understanding of the underlying principles of gradient boosting.

## Overview

The provided Python code implements two main classes:

* **`XGBoost`**: This class represents the core XGBoost algorithm. It handles the training of an ensemble of boosted trees. Key features include:
    * Initialization with customizable hyperparameters (learning rate, maximum depth, subsample, etc.).
    * Fitting the model to training data by iteratively adding new trees that correct the errors of the previous ones.
    * Row sampling during tree training to improve generalization.
    * Predicting raw scores based on the ensemble of trained trees.

* **`XGBoostSigmoid`**: This class is a wrapper around the `XGBoost` class specifically designed for binary classification tasks. It incorporates a sigmoid activation function and utilizes the binary cross-entropy loss function. Key features include:
    * Training the underlying `XGBoost` model.
    * Predicting probabilities using the sigmoid function on the raw output of the `XGBoost` model.
    * Option to predict binary labels based on a specified threshold (default is 0.5).
    * A `score` method to calculate the accuracy of the predictions.

Additionally, the code includes:

* **`BoostedTree`**: Represents a single decision tree within the boosted ensemble. It handles finding the best splits based on gradient and Hessian information and recursively builds the tree structure.
* **`SigmoidBinaryCrossEntropyObjective`**: Implements the sigmoid function and the binary cross-entropy loss function, along with methods to calculate the gradients and Hessians required for the XGBoost training process.

## Results

The image shows a submission to the Kaggle competition using this custom XGBoost implementation. The public score achieved is **0.78468**. This indicates the accuracy of the model's predictions on a portion of the test dataset.

**Key takeaways from the result:**

* The custom XGBoost implementation is functional and capable of achieving a competitive score on the Titanic dataset.
* Further hyperparameter tuning and feature engineering could potentially improve the model's performance.

## How to Use

To use this code:

1.  Ensure you have the necessary libraries installed (`numpy`, `pandas`, `math`).
2.  Prepare your training and testing data, ensuring it is preprocessed appropriately for a machine learning model.
3.  Instantiate the `XGBoostSigmoid` class with your desired hyperparameters.
4.  Train the model using the `train` method, providing your training features and labels, along with the number of boosting rounds.
5.  Make predictions on your test data using the `predict` method. You can choose to get probability scores or binary labels.
6.  Evaluate the model's performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score).

## Further Improvements

Potential areas for improvement include:

* **Hyperparameter Tuning:** Experimenting with different values for hyperparameters such as learning rate, maximum depth, regularization terms, and subsampling ratios can significantly impact performance. Techniques like grid search or randomized search can be used for this purpose.
* **Feature Engineering:** Creating new features from the existing data (e.g., combining family size, extracting titles from names) can provide the model with more informative signals.
* **Cross-Validation:** Implementing cross-validation during training can provide a more robust estimate of the model's generalization performance and help in selecting better hyperparameters.
* **Early Stopping:** Monitoring the model's performance on a validation set during training and stopping early when performance plateaus can prevent overfitting and reduce training time.

This project serves as a valuable exercise in understanding the inner workings of the XGBoost algorithm and its application to a classic machine learning problem.