---
title:  "Introduction to Neural Networks (OMSCS Deep Learning Course Mote 2)"
date:   2021-02-10
author_profile: true
comments: true
classes: wide
---

Optimization of Deep Neural Networks
=====
Background
    - Backpropagation and automatic differentiation allows us to optmize any function composed of differentiable blocks
        - No need to modify the learning algorithm
        - The complexity of the function is only limited by **computation and memory**
        - Depth is important
            - Structure the model to represent an inherently compositional world
            - Theoretical evidence that it leads to parameter efficiency
            - Dimensionality reduction

Design decisions:
    - Architecture
        - What modules/layers?
        - How they connect?
        - Domain Knoweldge for bias
        - Example (Depend on applications):
            - Fully Connected Neural Network
            - Convolutional Neural Networks
            - Recurrent Neural Network
    - Data considerations
        - pre-process
        - normalize
        - augment data by adding noise or other perturbations
    - Training and Optimization
        - What optimizer should we use?
        - Different weight updates depending on the gradients
        - How should we initialize the weights
        - What regularizers should we use?
        - What loss function is appropriate?
    - Machine Learning Considerations
        - Trade-off between model capacity and amount of data
        - Adding appropriate biases based on knowledge domain

Architectural Consierations
    - Guided by the type of data used and its characteristics
    - Use the flow of gradients to analyze layers
    - Combination of linear and non-linear layers
        - combination of only linear layers has same representational power as one linear layer
    - Non linear layers are crucial
        - Composition of non-linear layers enables complex transformations of the data
    - Aspects for analyzing
        - Min/Max
        - Correspondence between input & output statistics
        - Gradients
            - Initialization (e.g. small value)
            - extremes
        - Computational Complexity
        - Example: Sigmoid
            - min 0, max 1
            - output always positive
            - saturates both side
            - gradients vanish at both end and always positive
        - Other example: Tanh, Rectified Linear Unit, Leaky ReLU
    - ReLU is most common but no one activation function is best
    - Sigmoid is avoided (normally)

Initialization
    - Determine how statistics of outputs behave
    - Determine how well gradients flow in the begining fo training
    - Limit use of full capacity of the model if improperly set up
    - Constant value initialization leads to a degenerate solution
    - Common approach:
        - Small normally distributed random numbers
            - Small weights as no prior importance
            - Keep the model within the linear region of most activation function
    - Deep networks (with many layers) are more sensitive to initialization
        - With a deep network, activations get smaller
            - STD reduces significantly
        - Too small leads to small updates. If gradients are small, no learning will occur and no improvement is possible
        - Larger initial leads to saturation
    - Maintain the variance at the output to be similar to that of input
        - Sample from uniform distribution based on number of input and number of output (fan-in, fan-out)
        - For tanh or similar, N(0,1) * (1/n_j)^0.5 (number of output nodes)
        - For ReLU, N(0,1) * (2/n_j) * 0.5

Preprocessing, Normalization and Augmentation
    - Data drives learning in deep learning
        - Understand data chracteristics
        - Understand the relationship between output statistics, layers, and gradients
    - Normalization improve gradientis flow and learning
        - substract mean and then divide by standard deviation
            - Can have a **layer** to normalize the data across the neural network
            - Get mean and variance for each dimention for the mini batch data
            - Add learnable parameters gamma (sacle) and beta (shift) and network can learn to not normalize
            - **Btach Normalize (BN) Layer**
                - Store mean/variance during training
                - Sufficient batch sizeds must be used
                - **torch.nn.SyncBatchNorm** to estimate batch statistics
        - Whitening through PCA
    - **Normalization is improtant before non-linearities**
        - very low/high value cause satuaration

Optimizers
    - Loss is non-convex as Deep Learning involves complex, compositional, non-linear functions
    - Issues
        - Local minima (sometimes not bad)
        - Noisy gradient estimates
        - saddle points (where the gradient of orthogonal directions are zero)
        - ill-conditioned loss surface
    - Use a subset of the data at each iteration to calculate the loss & gradients
        - Unbiased estimator but can have high variance
        - noisy steps/convergece become slower
    - **Gradient Descent**
        - Intuition: a ball rolling down loss surface and use momentum to pass flat surface
        - Update velocity
            - Velocity term is an exponential moving average of the gradient  
        - Update weight
        - Generalize SGD (Stochastic gradient descent)
        - Go along velocity first and then calculate gradient at new point
    - Use **Jacobians** to get the information on curvature of the loss surface
    - **SGD**
        - Conditional number is the ratio of the largest and smallest eigenvalue
        - SGD makes big steps with higher ratio and smaller steps in other dimension
        - Second -order optimization methods divide steps by curvature but expensive to compute 
        - SGD can get the needed result but with much more tuning
        - Have a dynamic learning rate for each weight
    - Other optimization algorithms
        - RMSProp
            - Keep a moving average of squared gradients
            - Does not saturate the learning rate
        - Adagrad
            - Use gradient statics to reduce learning rate
            - Sum up gradients over iterations
            - Directions with high curvature will have higher gradients and learning rate will reduce
        - Adam
            - Combine RMSProp and Adagrad  
            - Maintain both first and second moment statistics for gradients
    - Behave differently depending on landscape
        - Overshooting
        - stagnating
    - Plain SGD + Momentum can generalize better than adaptive methods but requre more tuning

Regularization
    - Standard regularization methods still apply
        - L1 Regularization
        - L1/L2 on weights
        - Elastics L1/L2
    - **Problem**
        - Network can rely on some strong features
        - Overfitting if not representative of test data
    - For each node, keep its output with probability p
        - Activations of deactivated nodes are essentially zero
    - Choose whether to mask out a particular node each iteration
    - Principle: Always try to have similar train and test time input/output distributions
    - Solution: During test time, scale outputs by p
        - The model should not rely too heavily on particular features
        - Training 2^n networks
            - Each configuration is a network
            - Most are trained with 1-2 mini batches of data

Data Augmentation
    - Performing a range of transformation to the data
        - "Increase" the data
        - Should not change meaning of the data
    - Random Crop
        - Take different crops during training
        - Be used during inference
    - Color Jitter
    - CowMix
    - Combine different ways together!!

The Process of Training Neural Networks
    - Monitor everything to understand what is going on
        - **Loss and Accuracy curves**
            - Or gradient statisics/characteristics
            - Other aspects of computation graph
    - **Training, Validation, Test Set**
        - cross-valiadtion (may not use that much in deep learning due to computational complexity)
    - Check the bounds of your loss function
        - With or without regularization
        - Check initial loss at small random weights
    - **Simplify the dataset to make sure your model can properly (over)-fit before applyting regularization**
    - Change in loss indicates spped of learning
        - Tiny loss change
            - learning rate too small
        - Loss turn to NaNs
            - learning rate too high
        - Divide by zero
        - Forgetting the log
        - autograd in pytorch could help
    - Over-fitting
        - Validatation loss/accuracy starts get worse after a while
    - Under-fitting
        - Validation loss very close to training loss or both are high
    - Training loss may be higher (average for epochs so training loss includes some itersation which is not good, so it may be higher than validation loss which is trained in the end of epoch)
    - Validation loss has no regualarization
    - Validation loss is measured at the end of an epoch
    - **hyper-parameters tunning**
        - Learning rate
        - Momentum
        - start with coarser search and perfrom finer search arond good values
    - Inter-dependence of Hyperparameterrs
        - Batch norm and dropout may  not be needed together
        - Learning rate should be changed proportionally to batch size
    - The goal is to optimize **loss function**
        - because it is differentiable
        - Metrics which are not differentiable
            - Accuracy
            - Precsion/recall
            - Other specialized metirics
        - Flat loss curves can be increasing accuracy
        - Precision/Recall curves represent the inherent tradeoff between number of positive predictions and correctness of predictions
        - TPR/True Positive Rate = TP/(TP+FN)
        - AUC/Area under the curve

Data Wrangling
    - Problem to solve
        - Missing value
        - Imbalance
        - Perfroamnce evaluation
        - etc...
    - Sampling: obtain data from the population
        - Simple Random Sampling
        - Stratified Random Sampling
    - Data Wrangling Process
        - Step 1: What is the population of interest? What sample are we evaluating? Is sample representative of population?  
        - Step 2: How do we cross validate to evaluate out result? How do we avoid overfitting and underfitting?
            - K-fold validation
            - Random Search vs Grid Search for Hyperparameters
            - Confirm hyperparameter range is sufficient
            - Temporal cross-validation considerations
            - Overfitting?
        - Step 3:
            - What prediction task do we care about? What is the evaluation criteria?       
        - Step 4: How do we create a reproducible pipeline?         
    - Create Synthetic minority case to help imbalance
    - Classification
        - Precision: TP/(TP+FP)
        - Recall: TP/(TP+FN)
        - Accuracy: (TP+TN)/(TP+TN+FP+FN)
    - Regression
        - Mean-squared error
        - Visually analyze errors

Objective Detection
    - Region CNN and Single Shot Detector can localize and classfiy objects in an image
    - Densly sample many boxes of differnet sizes at differnet anchor locations in the images
    - Goal
        - Classify a proposed box into foreground
    - IoU
        - Intersection over union
    - Proposed box is assigned a ground truth of label of
        - Foreground, if IoU with ground truth box>0.5
        - Background (much more than foreground box)
    - Evaluate
        - Cross Entropy
        - Focal Loss
    - Cross Validation Done Right in Class Imbalanced Settings
    - Cross validation can go wrong so need to adjust for each problem

Data Wrangling Best Practices
    - Clearly define your population sample
    - Understand the representativeness of your sample
    - Cross valication can go wrong. Understand relevant problem and task in practice
    - Know the prediction task of interest
    - Incorporate model checks and evaluate multiple predictive performance metrics

Data Preparation
    - Clean
        - Missing value
            - Missing at completely/sometimes/not a random
        - Numerical
            - mean, mode, most frequent, zero, constant
        - Categorical
            - hot-deck imputation, k-Nearest, Neightbors, deep-learned embeddings
    - Transform
        - Image
            - Color conversion
        - Text
            - Index, Bag of words, TF-IDF, Embedding
    - Pre-Process
        - Format your data based on the type of model
        - Improve numerical stability
        - Gaussian error distribution

Managing Bias
    - Anti-classification
    - Classification parity
    - Calibration
    





    




        




    