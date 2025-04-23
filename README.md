# Basic Multi-Layer Perceptron (MLP) for Wine Quality Classification

## Introduction

This project involves building a basic Multi-Layer Perceptron (MLP) from scratch using NumPy to classify wine quality based on physicochemical properties. The dataset used contains characteristics such as acidity, pH, and alcohol content, along with the quality rating of the wines. This project is structured as both a theoretical exercise and a practical application involving the Wine Quality Dataset.

## Project Overview

The assignment consists of two parts: theoretical exercises and practical tasks. The theoretical exercises focus on concepts related to Artificial Neural Networks (ANN), while the practical task involves implementing an MLP to classify wines based on their attributes.

### Theoretical Exercises

- **Logistic Regression vs Perceptron**: The project explores why Logistic Regression is often preferred over a classical Perceptron, discussing how the Perceptron can be tweaked to behave like a Logistic Regression classifier.
  
- **Cross-Entropy vs Mean Squared Error**: The difference between these loss functions is explained, with a focus on their application to classification problems. Cross-Entropy is generally better for classification tasks due to its ability to model probabilities more accurately.

- **Weight Initialization**: Different weight initialization methods, such as He and Xavier initialization, are discussed. Proper weight initialization is crucial for the successful training of neural networks.

- **Activation Functions**: A detailed comparison of ReLU and sigmoid activation functions is made, highlighting the benefits and drawbacks of each.

- **Sparse Connectivity & Pruning Techniques**: Sparse connectivity and its impact on model performance is examined, with a focus on pruning techniques like optimal brain damage and sensitivity-based pruning.

### Practical Exercise

#### Problem Description

The task is to implement a Multi-Layer Perceptron (MLP) from scratch using NumPy, utilizing the **Wine Quality Dataset** to classify wines based on their physicochemical properties. The dataset consists of two types of wine: red and white. For this project, we focus on the red wine dataset.

#### Dataset Description

The **Wine Quality Dataset** contains several attributes of wine samples:

- **Fixed acidity**
- **Volatile acidity**
- **Citric acid**
- **Residual sugar**
- **Chlorides**
- **Free sulfur dioxide**
- **Total sulfur dioxide**
- **Density**
- **pH**
- **Sulphates**
- **Alcohol**
- **Quality** (score between 0 and 10)

#### Steps Involved

1. **Data Loading**: 
   - The red wine dataset is loaded using NumPy from the provided CSV file.
   
2. **Data Preprocessing**: 
   - The dataset is preprocessed by normalizing or standardizing the features. The dataset is split into training and testing sets with a ratio of 80\% training data and 20\% testing data.

3. **Model Implementation**: 
   - A simple MLP architecture is implemented, starting with one hidden layer. The input nodes correspond to the attributes of the wine, while the output node corresponds to the quality score.

4. **Training**: 
   - The model is trained using the backpropagation algorithm to adjust the weights iteratively. The performance of the model is evaluated across different learning rates (from 1e-8 to 10).

5. **Evaluation**: 
   - The model's performance is evaluated on the test set using metrics such as accuracy, precision, recall, and F1-score.

6. **Analysis**: 
   - The results are analyzed, focusing on the convergence behavior of the model, areas of improvement, and the effects of different learning rates.

7. **Activation Functions**: 
   - Ten different activation functions are chosen, and their distributions are plotted. Additionally, the derivatives of these activation functions are plotted and analyzed.

#### Code Implementation

The following sections detail the Python code used to implement the MLP and the associated steps:

```python
# Loading the dataset
import numpy as np

# Load dataset (example path)
data = np.loadtxt('winequality-red.csv', delimiter=',')

# Separate features and labels
X = data[:, :-1]  # Features (attributes)
y = data[:, -1]   # Labels (quality)

# Data Preprocessing (Normalization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Implement MLP
# [Insert MLP code here]
```

#### Evaluation Metrics

- **Accuracy**: The percentage of correct predictions.
- **Precision**: The percentage of relevant instances among the retrieved instances.
- **Recall**: The percentage of relevant instances that have been retrieved.
- **F1-Score**: The harmonic mean of precision and recall.

### Results

- The results of the model are analyzed in terms of loss reduction across different learning rates.
- A plot of the loss reduction over epochs for different learning rates is generated.

```python
import matplotlib.pyplot as plt

# Example: Plotting loss reduction
plt.plot(learning_rates, loss_values)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Loss Reduction for Different Learning Rates')
plt.show()
```

### Activation Functions & Their Distributions

Ten activation functions are selected, and their distributions are visualized. These functions include:

- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Softmax
- Swish
- ELU
- SELU
- GELU
- Hard Sigmoid

Each of these functions and their derivatives is plotted to analyze their behaviors.

## Conclusion

This project demonstrated the practical application of an MLP for wine quality classification. The implemented model used various techniques, including backpropagation and different learning rates, to optimize performance. The analysis of activation functions and pruning techniques provides further insights into improving neural network models.

## References

- Wine Quality Dataset: [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
