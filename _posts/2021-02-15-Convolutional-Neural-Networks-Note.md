---
title:  "Convolutional Neural Networks (OMSCS Deep Learning Course Note 3)"
date:   2021-02-15
author_profile: true
comments: true
classes: wide
---

Convolutional Neural Networks
=====
Introduction
    - convolution layer can take any input 3D tensor and output another similarly-shaped output
    - e.g. RGB image to gray
    - convolution layers can be combined with non-linear and pooling layers whcih reduce the dimentionality of the data
    - Backpropagation and automatic differentiation allow us to optimize any function composed of different blocks
        - no need to change learning algorithm
        - the compl

Process Example
    - Image
    - Conv + Non-Linear Layer
    - Pooling Layer
    - Conv + Non-Linear Layer
    - Fully Connected Layers

The connectivity in linear layers does not always make sense
    - More parameters lead to more data
    - **Not necessary**

Image features are spatially localized
    - smaller features repeated across the image
        - Edges
        - Color
        - Motifs

Each node only receive input from K_1*K_2 window
    - receptive field
    - Advantage
        - Reduece parameters to (K_1*K_2+1)*N
        - Explicitly maintain spatial information

nodes in different locations can share features
    - No reason to think same features can't appear elsewhere
    - Use same weights/parameters in computation graph
    - Advantage
        - Reduce to (K_1*K_2+1)
        - Explicitly maintain spatial information

Learn many such features for this one layer
    - Weights are not shared across different feature extractors
    - Parameters (K_1*K_2+1)*M, where M is number of features we want to learn

Convolution Example Process
    - Image
    - Kernel/Filter
    - Output/Filter/feature map

**Convolution Layer**
    - Initialize kernel
    - Parameters plus a bias term per filter
    - Intuitive Explanation
        - Filp kernel
        - Stride along images
        - Generate output
    - Convolution
        - Start at end of kernal and move back
    - Cross-correlation
        - Start in the beginning and move forward

Why Convolution?
    - just simple linear operations and just a linear layer with small receptive field
    - duality between them during backpropagation
    - convolution have various mathematical properties 
    - Historically how it was inspired

Input  and Output Sizes
    - Parameters
        - Number of channels in input image
        - Number of channels produced by the convolution
        - kernel size
        - stride
        - padding
        - padding mode
    - Out size of vanilla convolution operations
        - (H-K_1+1)*(W-k_2+1)
    - We can pad the image to make the output the same size
        - zeros
        - padding ofen refers to pixels add to one size/p=1
            - output size (H+2-K_1+1)*(W+2-K_2+1)
    - Stride
        - Move filter along the image using larger steps
            - potentially result in loss of information
        - With stride = 2
            - ((H-K_1)/2+1)*((H-K_2)/2+1)
    - In reality, there have more than one channel
        - Image has three channels RGB
            - Kernel can be 3-channel kernels
        - Can have multiple layers per layer
    - Number of filters with input H*W*N and kernel with K_1*K_2*m
        - N is 3 for images and m is the number of filters
        - m*(K_1*K_2*N+1)
        - each kernel has a bias term

Steps to vectorize this operation
    - Step 1: Layout image patches in vector form/Im2cool
    - Step 2: Multiple patches by kernel

**Pooling Layer**
    - pooling operations
        - down sample
    - parameters
        - kernel size
        - stride
        - padding
    - Can use any differentiable function
    - Example
        - max pooling/no parameters needed
        - average pooling

Combination Convolution and Pooling layers
    - The combination adds some **invaraince** to translation of the features
        - If feature translated a little bit, output values still remain the same
    - Convolution by itself has the property **equivariance**
        - If feature translated a little bit, output values move by the same translation

Backwards pass for Convolution Layer
    - It is instructive to calculate the backwards pass of a convolution layer
    - gradient for passing back
    - gradienst for weight update
    - 1 padding, 1 channel and 1 kernel to make output the same size
