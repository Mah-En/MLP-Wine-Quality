
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
   - The dataset is preprocessed by normalizing or standardizing the features. The dataset is split into training and testing sets with a ratio of 80% training data and 20% testing data.

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

#### Selected Images and Explanations

### Image 1: Plot of Loss Reduction (Plot 28)
![Plot 28](sandbox:/mnt/data/plot_28.png)

This image shows the loss reduction over training iterations for a specific learning rate. It is critical for visualizing how well the model is converging during training, helping to assess the effectiveness of the optimization process.

### Image 2: Training Accuracy vs Epochs (Plot 29)
![Plot 29](sandbox:/mnt/data/plot_29.png)

This plot displays the accuracy of the model over each epoch. Tracking accuracy during training helps in determining whether the model is learning the patterns effectively or if adjustments are necessary.

### Image 3: Activation Function Distribution (Plot 30)
![Plot 30](sandbox:/mnt/data/plot_30.png)

This image presents the distribution of one of the activation functions used in the model. Understanding activation function behavior is crucial for selecting the right one for the problem and improving the model's ability to learn complex patterns.

### Image 4: Activation Function Derivative (Plot 31)
![Plot 31](sandbox:/mnt/data/plot_31.png)

The derivative of the activation function is shown here. This is essential for understanding how the weights are adjusted during backpropagation, affecting the learning process.

### Image 5: Model Performance with Different Learning Rates (Plot 32)
![Plot 32](sandbox:/mnt/data/plot_32.png)

This image compares model performance using different learning rates. It helps identify the optimal learning rate for faster convergence and better model accuracy.

### Image 6: F1-Score Across Epochs (Plot 33)
![Plot 33](sandbox:/mnt/data/plot_33.png)

The F1-score is plotted here, giving a balanced measure of the model's precision and recall. This metric is important for evaluating classification performance, especially when there is an imbalance in the dataset.

## Conclusion

This project demonstrated the practical application of an MLP for wine quality classification. The implemented model used various techniques, including backpropagation and different learning rates, to optimize performance. The analysis of activation functions and pruning techniques provides further insights into improving neural network models.

## References

- Wine Quality Dataset: [Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
