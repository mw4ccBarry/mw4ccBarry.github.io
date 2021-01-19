---
title:  "Tree based methods review"
date:   2020-6-27
author_profile: true
comments: true
classes: wide
header:
  teaser: /assets/images/decision_tree.png
---

Overview
==========
**Segment the predictor space** into a number of regions and use the **mean or mode of the observations in the region** for prediction. The segment/spliting rules can be summarized in a tree so the method is named as decision tree methods.


**Advantages:**
- Interpretability
- Visualization
- Mirror people's thinking process
- Handle qualitative predictors


**Disadvantages:**
- non-robust
- performance


**Some concepts for tree:** terminal node, leaf, internal node, branch, root node, splitting, pruning


**Widely used algorithms:** 
- ID3 (categorical feature and target, information gain), 
- C4.5 (can have numerical variables, trees to set of rules), 
- CART (used by sklearn and H2O, should be good for both numerical and categorical variable while sklearn could not directly handle categorical variable, use Gini index by default), 
- MARS (capture the nonlinear relationships in the data by assessing cutpoints (knots) similar to step functions. it is not decision tree actually but is actually created a regression tree like model)


Regression Tree
==========
**Build Tree (Splitting feature space)**
The book introduce the recursive binary splitting which is a top-down, greedy approach to build the decision tree. For each stratification, we need to find the predictor J and cutpoint s to split the feature space as below,

$$R_1(j,s) = \{ X|X_j < s\}\ and\ R_2(j,s) = \{ X|X_j \geq s\} $$

so that the splitted feature space can minimize the below function,

$$\sum_{x_i{\in}R_1(j,s)}(y_i - \hat{y}_{R_1})^{2} + \sum_{x_i{\in}R_2(j,s)}(y_i - \hat{y}_{R_2})^{2}$$	

With the decistion tree built, the predict/query to the model is just using the rules for splitting the regions and get the mean for the observation. 

**Note:** The implementation might be different. When I create the decision tree learner, I am caculating the correlation between X and y to find the J and use median for splitting value.

**Tree Pruning**


Tree pruning will help overfitting and improve interpretability. Intuitively, we can grow a large tree and prune it back to get subtree by getting the subtree with the lowest test error rate. This is not feasible due to feasibility. 

|smaller tree|larger tree   |
|---|---|
|higher bias|lower bias|
|lower variance|higher variance|
|better interpretation|worse interpretation|

Thus, the book introduce the method **complexity pruning** with the **Tree Complexity Penalty**. Instead of minimize sum of squared Residuals, we will minimize that plus the function with number of terminal nodes as the below function. The larger the tuning parameter the simpler the tree we will get. For the tree and each sub-tree, we can have the tuning parameter for each of them. Then run the cross valiadation to choose the tuning parameter and then the sub tree. The function for each sub tree is below:

$${\sum_{m=1}^{|T|}}{\sum_{i:x_i \in R_m}(y_i - \hat{y}_{R_m})^2 + \alpha{|T|}}$$

**Summary**
- Recursive binary splitting to grow the tree with training data
- Apply cost complexity to prune the tree with all the tuning parameter for each best subtrees
- Coss validation to determine tuning parameter and decide the subtree

**Regression Tree vs Linear Models**
- Choose which one to use by comparing performance
- Linear vs Non Linear
- In my opinion, if we want to interprete some product actions, tree regression is preferred. If we want to estiamte the contribution like sizing the impact, linear regression may be better.


Classification Tree
=========
Intuitively, similar to the regression tree, the task of growing classification tree could also utilize recursive binary splitting to optimize the classification error rate defined as the following function. 

$$E = 1 - \max_k(\hat{p}_{mk})$$

There are 2 other preferable metrics, Gini index and Entropy. Note: following Gini index is purity so that it will be substract from 1 to get impurity.

 $$G=\sum_{k=1}^K\hat{p}_{mk}(1-\hat{p}_{mk})$$

Belowe is Entropy:

 $$D=-\sum_{k=1}^K\hat{p}_{mk}\log\hat{p}_{mk}$$

Below is the information gain using Entropy:

 $$Infomation\ Gain(split) = Entropy(parent) - (weighted) Average\ Entropy(Children)$$

Gini index measure the total variance across K classes so that it measures the node purity which suggests the node contains predominantly observations from one class.Gini index and Entropy is better than classification error in building the tree because classification error is **not sensitive enough** so that the tree may not grow as expected. Classification error can be used for pruning better than other metrics as it will be better align with our classification problem. 

In the work of [Theoretical comparison between the gini index and information gain criteria](https://www.unine.ch/files/live/sites/imi/files/shared/documents/papers/Gini_index_fulltext.pdf), the findings are below:
- It only matters in 2% of the cases whether you use gini impurity or entropy.
- Entropy might be a little slower to compute (because it makes use of the logarithm).


Bagging
========
Introduce bootstrap aggregation or bagging to reduce the high variance introduced by the tree method. We do bootstrap by taking repeated sample from the single training data sets as described as the following function,

$$\hat{f}_{bag}x=\frac{1}{B}\sum_{b=1}^B\hat{f}^{*b}(x)$$

Basically, the method constructed B regression trees using B bootstrapped training sets and average the predictions. The trees are grown deep without pruned which have high variance and low bias. The method could get average for regression and majoriry vote for classification. 


Each bagged tree makes use of 2/3 of the obvervations. For the given bagged tree, the method can use the remaining ovservations to get out-of-bag observations. We can then average those predicted responses and evaluate the OOB MSE or OOB classification error. With B to be large, the OOB error is equivalent to leave-one-out cross-validation error.

The prediction accuracy is improved at the expense of interpretability. For bagging, the feature importance can be obtained with adding up the total amount that the Gini index is decreased by splits over a given predictor, averaged over all B trees. Same for using RSS for regression trees.

**Issue:** All bagged trees are similar and highly correlated as all trees will use the strong predictor in the top split. 

Random Forest
=======
Random forest improved bagging by reduce the correlation among trees by limiting the number of features used in each split. The feature importance of random forest is evaluated as the decrease in node impurity weighted by the probability of reaching that node.A random sample of m predictors is chosen by each split.Typically, m is chosed as,

$$m\approx\sqrt{p}$$

Random forest is just the bagging method when m = p. Random forest helps a lot when the data includes a large number of correlated predictors. Accuracy of a single tree is ~50%. 

Boosting
======
Trees are grown sequentially as each tree is grown using information from previously grown tree. Boosting intends to build smaller trees and learns slowly. Given the current model, we fit a decisiton trees to the reiduals from the model. The new tree is added into the fitted function to update the residuals. 

Tuning parameters include:
- number of trees B
- shrinkage parameter
- number d of splits in each tree

Algorithm:
- Start with fitting Y
- For 1,2,...B
  - Fit tree with d splits to the updated residuals with the shrinkage parameter
  - Add new tree to the model
  - update residuals
- Output the boosted model with shrinkage factor