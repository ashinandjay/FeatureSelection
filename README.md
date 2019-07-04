# Feature Extraction and Selection Tool
*A feature extraction and selection Python package for DNA, RNA, Protein sequence data.*

## Table of Contents

### * [Installation](#Installation)
### * [Data Preparation](#Data-Preparation)

### * [Feature Selection](#Feature-Selection)
### * [Feature Reduction](#Feature-Reduction)
### * [Feature Evaluation](#Feature-Evaluation)
### * [Feature Evaluation Neural Network](#Feature-Evaluation-Neural-Network)
### * [Output](#Output)

## Installation

The package is developed using Python 3(Python Version 3.0 or above) and it can be run on Linux operatinig system. We strongly recommand user to install [Ancaonda Python 3.7 or above version](https://www.anaconda.com/distribution/) to avoid installing other packages.

After installaing Anaconda, the following packages need to be installed:
1. xgboost
2. skrebate
3. lightgbm

The source code is freely available at: https://github.com/ashinandjay/FeatureSelection

To install our tool, first download the zip file manually from github, or use the code below in Unix:
```{r,engine='bash',eval=FALSE, download}
cd your_folder_path
wget https://github.com/ashinandjay/FeatureSelection/archive/master.zip
```
Unzip the file:
```{r,engine='bash',eval=FALSE, unzip}
unzip master.zip
```

## Data Preparation

The DNA, RNA or protein sequence data (FASTA format) and their labels (txt format) are required for using our feature selection tool.

## Feature Selection

Our Feature Selection tool contains 20 supervised selection methods.

Feature Selection Method | Feature Selection Number
------------------------ | -------------------------
Lasso | 1
Elastic Net | 2
L1-SVM | 3
CHI2 | 4
Pearson Correlation | 5
ExtraTree | 6
XGBoost | 7
SVM-RFE | 8
LOG-RFE | 9
Mutual Information | 10
Minimum Redundancy Maximum Relevance | 11
Joint Mutual Information | 12
Maximum-Relevance-Maximum-Distance | 13
ReliefF | 14
Trace Ratio | 15
Gini index | 16
SPEC | 17
Fisher Score | 18
T Score | 19
Information Gain  | 20

For using our Feacture Selection Tool, Four inputs are required: 
1. Feauture selection number (See the table above)
2. Number of feature to select (how many number of feature you want)
3. Feature Vectors (Feature extraction output file)
4. Label Vectors (labels for sequencing)

Run Feature_Selection.py:
```{r,engine='bash',eval=FALSE}
Feature_Selection.py [Feauture selection number] [Number of feature to select] [Feature Vectors] [Label Vectors]
```

Example: Using **Lasso** method to select **3** features
```{r,engine='bash',eval=FALSE}
Feature_Selection.py 1 3 Feaute_Vectors.tsv label.txt
```

## Feature Reduction
Our Feature Reduction tool contains 13 unsupervised dimensionality reduction methods.

Feature Reduction Method | Feature Reduction Number
------------------------ | -------------------------
K-means | 1
T-SNE   | 2
Principal Component Analysis | 3
Kernel PCA | 4
Locally-linear embedding | 5
Singular Value Decomposition | 6
Non-negative matrix factorization | 7
Multi-dimensional Scaling | 8
Independent Component Analysis | 9
Factor Analysis | 10
Agglomerate Feature | 11
Gaussian random projection | 12
Sparse random projection | 13

For using our Feacture Reduction Tool, Three inputs are required: 
1. Feauture Reduction number (See the table above)
2. Number of Clusters (how many number of Clusters you want)
3. Feature Vectors (Feature extraction output file)

Run Feature_Reduction.py:
```{r,engine='bash',eval=FALSE}
Feature_Reduction.py [Feauture Reduction number] [Number of Clusters to select] [Feature Vectors]
```

Example: Using **PCA** method to select **3** clusters
```{r,engine='bash',eval=FALSE}
Feature_Reduction.py 3 3 Feaute_Vectors.tsv

## Feature Reduction
Feature selection method can be evaluated using 10 classifiers.

Index | Classifier Names
----- | ----------------
1 | SVM
2 | KNN
3 | RandomForest
4 | LightGBM
5| xgboost
6 | AdaBoost
7 | Bagging
8| ExtraTree
9 | Gaussian Naive Bayes
10 | Gradient Boosting

Run Feature_Evaluation.py:
```{r,engine='bash',eval=FALSE}
Feature_Evaluation.py [Feauture selection output] [Label Vectors]
```
Example: evaluating Lasso selection method
```{r,engine='bash',eval=FALSE}
Feature_Evaluation.py Lasso.csv label.txt
