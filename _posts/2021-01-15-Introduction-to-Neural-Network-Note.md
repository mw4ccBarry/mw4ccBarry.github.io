---
title:  "Introduction to Neural Networks (OmSCS Deep Learning Course Mote 1)"
date:   2021-01-15
author_profile: true
comments: true
classes: wide
---

Linear Classiferers and Gradient Descent
=====

**Machine Learning** is the study of algorithms that:
- Improve their performance
- on some task(s)
- Based on expereince (typically data)
- **Supervised/Unsupervised/Reinforcement**:
    - Supervised Learning (Input Data and Label to get Learning Output)
    - Unsupervised Learning (Input Data and **NO** Label to get Learning Output)
    - Reinforcement Learning (Supervision in form of reward)
- **Non-Parametric Model/Paremetric Model**:
    - Non-parametric: No explict model for the function
        - Nearest Neighbor, Decision Tree
    - Paremetric: Explicitly model the function in the form of a parametrized function
        - Logistic Regression, Neural Network


*Keyword*: Feature Engineering, High Dimention data to Low Dimention data   

**Components of a Parametric Learning Algorithm**
- Input/Representation
- Functional Form of the model (with parameters)
- Performance measure to improve 
    - Loss or objective function
        - Score and probability: Score is hard to inteprete while probability is better and relate to probabilistic view of machine learning.
        - Deep learning use **softmax function** to convert score to probability
    - We need a performance measure to optimize
        - penalize model for being wrong
        - allow us to modify the model to reduce penalty
    - Empirical risk minimization
        - reduce the loss over training dataset
        - average the loss over training data
    - Multiclass Loss - Hinge Loss
    - **Cross-Entropy**
        - The distance between two probability distributions (model outputs and truth)
        - Maximum likelihood estimation
            - choose probabilities to maximize the liklihood of the observed data
    - Loss function for Regression
        - L1, L2, Logistic
    - *Regularization term* to the loss function
- Algorithm for finding best parameters
    - Optimization algorithm
        - Given a model and loss function, finding best set of weights is a search problem
        - Search problem/Find the best combination of weights that minimizes loss function
            - Random search
            - Genetic algorithms
            - *Gradient-based optimization*
    - Gradient Descent
        - Find the steepest descent direction through derivative/gradient (steepest descent is the negative gradient)
        - Take partial derivative of loss function with repect to that parameters
        - **Steps**
            - Model
            - Loss function
            - Partial derivative for each parameter
            - Update the parameters
            - Add learning rate to prevent big step when update parameters
        - Full Batch/Mini-Batch
            - Get mini-batch, compute loss, compute derivatives and take a set
        - How to do partial derivative
            - Manual/Symbolic/Numerical/Automatic Differentiation
        - Note
            - Learning rate has to be reduced
            - Local minima
            - Local minima is not bad :)
    - Vector and Matrix related to Gadient Descent
        - Size of the partial derivative of a scaler (Loss) and Matrix (Weight)
        - Jacobian is a matrix as the same size of Weight
        - Jacobian is complcated to become tensor so flatten to a vector of derivatives


*Keyword*: Linear Regression and Classification

Great resource from StatQuest for [Gradient Descent!](https://www.youtube.com/watch?v=sDv4f4s2SB8&feature=youtu.be) 

Intro to Deep Learning
====

What is Deep (Machine) Learning?
- Representation Learning
- Neural Networks
- Deep Unsupervised/Reinforcement/Structured/... Learning

Neural Network View of a Linear Classifier
- Linear Classifier
    - Input
    - Model/Input Function
    - Loss function
- A simple neural network has similar structure as linear classifier
    - A neuron takes inut from other neurons
    - Inputs are summed in a weighted manner (weighted sum)
    - plus bias terms
    - *The output of a neuron can be modelated by a non-linear function (sigmoid)

**Hierarchical Compositionality**
- Cascade of Non-Linear
- Multiple Layers of representation
**End-to-End Learning**
- Learning Representations
- Goal Driven
- Learning Feature Extraction
**Distributed Representations**
- No single neuron encodes everything
- grups of neuron works together

Ways to extract features/function, Given a library of features
- Linear Combinations (Boosting, Kernels...)
- Compositions (Deep Learning, Grammer Models...)

**Multi-class classifier**
- Multiple neurons connected to the same input
- Each output node outputs the score for a class
- fully connected layers/linear projection layer

**Notations**:
- Each input/output is a **neuron/node**
- Linear classifier is **fully connected layer**
- Connections are represented as **edges**
- Output of a neuron is **activation**
- Viewed as **graph**
- Stack multiple layers together
    - Input to second layer is output of first layer
    - **Hidden Layer**
    - The hidden layer corresponds to adding another weight matrix
    - More and more layers in large/deep networks
    - Can represent **any function**

**Computation Graphs**
Intuition
- Use any type of differentiable function we want with the end to be loss function
- Connect things in a way to reflect the complex composition problem
Directed acyclic graph (DAG)
- Module must be differntiable
- Training problem compute one module at a time

**Backprogation**
The training algorithm need:
- Calcualte the current model's output (Forward pass)
- Calculate the gradients for each module (Backward pass)
Backward Pass is a recursive algorithm
- Start with loss function
- Progress back through modules

*Step1*

- **Compute loss on Mini-Batch: Forward Pass**
- Store the intermediate output of all layers for computing gradients
- Calculate the gradients of the loss with respect to the module's parameters
- The gradient of the loss with respect to the module's input is also passed
- Compute the local gradients, just the dirivative of our functions with respect to its parameters and inputs
- Get the gradient of loss w.r.t. inputs/weights

*Step2*

- **Compute Gradient wrt parameters: Backward Pass**

*Step3*

- **Use gradient to update all parameters at the end**


**Backpropagation and Automatic Differentiation**
- Backpropagation can be applied to any directed acyclic graph (DAG)
- Given the ordering, we can iterate from the last module backwards applying the chain rule
    - store the gradiet outputs for each node for efficient computation
- reverse-mode automatic differentiation
- **Computation = Graph**
    - Input = Data + Parameters
    - Output = Loss
    - Scheduling = Topological ordering
    - Auto-Diff
        - Implementing chain-rule on computation graphs
    - Find partial derivative of output with respect to all intermediate variables
    - **Patterns of Gradient Flow**
        - Multiplication is a gradient switcher
        - Max select which path to push the gradient through
    - If gradients do now flow backward, stop or slow to learn
    - Explicitly store computation graph in memory and corrsponding gradient functions
    - Nodes brokend down to basic primitive computations (addition, multiplication, log) for which corresponding derivative is known
    - We can also do forward mode automatic differentiation starting from inputs and propagate gradients forward. Most cases have large inputs/images and outputs/loss are small.

**Note1: Back-propagation uses the dynamically built graph with torch.autograd**


**Note2: Computation graphs are not limited to mathmatical functions but can have control flows through differentiable programming**


**Note3: Check Matrix Calculas for Deep Learning for tensor and derivatives for backpropagation**
