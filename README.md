# **ALLFEAUTRE** An intergrated python package for DNA, RNA and Protein sequecing data analysis
*Include Feature extraction, Feature selection, Dimensionality Reduction, Models Construction for sequencing data.*

## Table of Contents

### * [Installation](#Installation)
### * [Data Preparation](#Data-Preparation)
### * [DNA Feature Extraction](#DNA-Feature-Extraction)
### * [RNA Feature Extraction](#RNA-Feature-Extraction)
### * [Protein Feature Extraction](#Protein-Feature-Extraction)
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

## DNA Feature Extraction

The tool includes 16 feature extraction methods for DNA sequencing data.

DNA Extraction Method | DNA Extraction Number
--------------------- | -------------------------
Kmer | 1
Reverse Compliment Kmer | 2
Pseudo dinucleotide composition | 3
Pseudo k-tuple nucleotide composition | 4
Dinucleotide-based auto covariance | 5
Dinucleotide-based cross covariance | 6
Dinucleotide-based auto-cross covariance | 7
Trinucleotide-based auto covariance | 8
Trinucleotide-based cross covariance | 9
Trinucleotide-based auto-cross covariance | 10
Nucleic acid composition | 11
Di-nucleotide composition| 12
Tri-nucleotide composition | 13
zcurve | 14
monoMonoKGap | 15
monoDiKGap | 16

DNA_Feature_Extraction require two inputs: DNA Extraction number and DNA sequencing data.

Run DNA_Feature_Extraction.py:
```{r,engine='bash',eval=FALSE}
DNA_Feature_Extraction.py [DNA Extraction number] [DNA sequencing data]
```

Example: Use **kmer** method to extract features from DNA sequencing data
```{r,engine='bash',eval=FALSE}
DNA_Feature_Extraction.py 1 DNA_sequencing.txt
```

## RNA Feature Extraction

The tool includes 12 feature extraction methods for RNA sequencing data.

RNA Extraction Method | RNA Extraction Number
--------------------- | -------------------------
Kmer | 1
Reverse Compliment Kmer | 2
Pseudo dinucleotide composition | 3
Dinucleotide-based auto covariance | 4
Dinucleotide-based cross covariance | 5
Dinucleotide-based auto-cross covariance | 6
Nucleic acid composition | 7
Di-nucleotide composition| 8
Tri-nucleotide composition | 9
zcurve | 10
monoMonoKGap | 11
monoDiKGap | 12

RNA_Feature_Extraction require two inputs: RNA Extraction number and RNA sequencing data.

Run RNA_Feature_Extraction.py:
```{r,engine='bash',eval=FALSE}
RNA_Feature_Extraction.py [RNA Extraction number] [RNA sequencing data]
```

Example: Use **kmer** method to extract features from RNA sequencing data
```{r,engine='bash',eval=FALSE}
RNA_Feature_Extraction.py 1 RNA_sequencing.txt
```

## Protein Feature Extraction

The tool includes 32 feature extraction methods for Protein sequencing data.

Protein Extraction Method | Protein Extraction Number
--------------------- | -------------------------
Amino acid composition | 1
Composition of k-spaced amino acid pairs | 2
Dipeptide composition | 3
Grouped dipeptide composition | 4
Grouped tripeptide composition  | 5
Cojoint triad | 6
k-spaced cojoint triad | 7
Composition| 8
Transition | 9
Distribution | 10
Encoding based on grouped weight | 11
Auto covariance| 12
Moran autocorrelation | 13
Geary autocorrelation | 14
Quasi-sequence-order | 15
Pseudo-amino acid composition | 16
Amphiphilic pseudo-amino acid composition | 17
Amino Acid Composition PSSM | 18
Dipeptide composition PSSM | 19
Pseudo PSSM | 20
Auto covariance PSSM | 21
Cross covariance PSSM | 22
Auto Cross covariance PSSM | 23
Bigram-PSSM | 24
AB-PSSM | 25
Secondary structure composition | 26 
Accessible surface area composition | 27
Torsional angles composition | 28
Torsional angles bigram | 29
Structural probabilities Bigram | 30
Torsional angles auto-covariance | 31
Structural probabilities auto-covariance | 32 

Protein_Feature_Extraction require two inputs: Protein Extraction number and Protein sequencing data.

Run Protein_Feature_Extraction.py:
```{r,engine='bash',eval=FALSE}
Protein_Feature_Extraction.py [Protein Extraction number] [Protein sequencing data]
```

Example: Use **Amino acid composition** method to extract features from Protein sequencing data
```{r,engine='bash',eval=FALSE}
Protein_Feature_Extraction.py 1 Protein_sequencing.txt
```
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
Feature_Selection.py 1 3 Feaute_Vectors.csv label.txt
```

## Feature Reduction
Our Feature Reduction tool contains 16 unsupervised dimensionality reduction methods.

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

Our Feature Reduction tool also contains 3 deep learning dimensionality reduction methods.

Feature Reduction Method | Feature Reduction Number
------------------------ | -------------------------
Autoencoder | 1
Gaussian Noise Autoencoder | 2
Variational Autoencoder | 3

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
Feature_Reduction.py 3 3 Feaute_Vectors.csv
```

## Feature Evaluation
Feature selection method can be evaluated using 10 classifiers. The classification accurcay comparison files (plot and table) will be generated in same folder of code.

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
```

## Feature Evaluation Neural Network
Feature selection also can be evaluated using 3 neural network classification methods.

Feature Evaluation NN | Feature Evaluation NN Number
--------------------- | -------------------------
Convolutional neural network | 1
Deep neural network   | 2
Recurrent neural network | 3
