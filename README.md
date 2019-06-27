# Feature Extraction and Selection Tool
*A feature extraction and selection Python package for DNA, RNA, Protein sequence data.*

## Table of Contents

### * [Installation](#Installation)
### * [Data Preparation](#Data-Preparation)

### * [Feature Selection](#Feature-Selection)
### * [Feature Clustering](#Feature-Clustering)
### * [Feature Evaluation](#Feature-Evaluation)
### * [Feature Evaluation NN](#Feature-Evaluation-NN)
### * [Output](#Output)

## Installation

The package is developed using Python 3(Python Version 3.0 or above) and it can be run on Linux operatinig system. We strongly recommand user to install [Ancaonda Python 3.7 or above version](https://www.anaconda.com/distribution/) to avoid installing other packages.

After installaing Anaconda, the following packages need to be installed:
1. xgboost
2. skrebate

The source code is freely available at: https://github.com/ashinandjay/FeatureSelection

To install our tool, first download the zip file manually from github, or use the code below in Unix:
```{r,engine='bash',eval=FALSE, download}
cd your_folder_path
wget https://github.com/ashinandjay/FeatureSelection.git
```
Unzip the file:
```{r,engine='bash',eval=FALSE, unzip}
unzip master.zip
```

## Data Preparation

The DNA, RNA or protein sequence data (FASTA format) and their labels (txt format) are required for using our feature selection tool.

## Feature Selection

Our feature selection tool contains 20 supervised selection methods.

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
MRMR | 11
JMI | 12
MRMD | 13
ReliefF | 14
Trace Ratio | 15
Gini index | 16
T Score | 17
SPEC | 18
Fisher Score | 19
LFDA | 20

For using our Feacture Selection Tool, Four input are required: 
1. Feauture selection number (See the table above)
2. Number of feature to select (how many number of feature you want)
3. Feature Vectors (Feature extraction output file)
4. Label Vectors (labels for sequencing)

Run Feature_Selection.py:
```{r,engine='bash',eval=FALSE}
Feature_Selection.py [Feauture selection number] [Number of feature to select] [Feature Vectors] [Label Vectors]
```

Example:using Lasso method to select 3 features
```{r,engine='bash',eval=FALSE}
Feature_Selection.py 1 3 Feaute_Vectors.tsv label.txt
```

## Feature Clustering

Our feature selection tool contains 20 supervised selection methods.
