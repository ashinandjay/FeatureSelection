# This code is used for feature clustering methods, 
# The output file will be generated in same folder of this code
# -------------------------------------------------------------

# load packages:
import sys,os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA

# find the path
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# choose the method
option = sys.argv[1]

# read number of clustering
selected_number = sys.argv[2]

# read feature vectors
feature = sys.argv[3]
feature_o = pd.read_csv(feature,delim_whitespace=True,index_col=0,float_precision='round_trip')

# Feature selection PCA
def pcaselection(k,X):
    X=np.array(X)
    pcaresult = PCA(n_components=k).fit_transform(X)
    np.savetxt("PCA_out.csv", pcaresult, delimiter=",")
    return None

# Feature selection kernelPCA
def kernelPCA(k,X):
    X=np.array(X)
    kpca=KernelPCA(n_components=k)
    kpcaresult=kpca.fit_transform(X)
    np.savetxt("KernelPCA_out.csv", kpcaresult, delimiter=",")
    return None

# Feature selection Locally-linear embedding
def lle(k,X):
    X=np.array(X)
    embedding = LocallyLinearEmbedding(n_components=k)
    lleresult = embedding.fit_transform(X)
    np.savetxt("LLE_out.csv", lleresult, delimiter=",")
    return None

# Feature selection SVD
def svd(k,X):
    X=np.array(X)
    SVDMethod = TruncatedSVD(n_components=k)
    svdresult=SVDMethod.fit_transform(X)
    np.savetxt("SVD_out.csv", svdresult, delimiter=",")
    return None

# Feature selection NMF
def nmf(k,X):
    X=np.array(X)
    NMFmodel=NMF(n_components=k)
    nmfresult= NMFmodel.fit_transform(X)
    np.savetxt("NMF_out.csv", nmfresult, delimiter=",")
    return None

# Feature selection MDS
def mds(k,X):
    MDSmodel = MDS(n_components=k)
    mdsresult = MDSmodel.fit_transform(X)
    np.savetxt("MDS_out.csv", mdsresult, delimiter=",")
    return None

# Feature selection ICA
def ICA(data,n_components):
    ica = FastICA(n_components=n_components)
    X_transformed = ica.fit_transform(data)
    np.savetxt("ICA_out.csv", X_transformed, delimiter=",")
    return None

# Feature selection FA
def FA(data,n_components):
    fa= FactorAnalysis(n_components=n_components)
    FAresult=fa.fit_transform(data)
    np.savetxt("FA_out.csv", FAresult, delimiter=",")
    return None

# main code

if(option == "1"):
    # Feature selection PCA
    pcaselection(selected_number,feature_o)
elif(option == "2"):
    # Feature selection KernelPCA
    kernelPCA(selected_number,feature_o)
elif(option == "3"):
    # Feature selection LLE
    lle(selected_number,feature_o)
elif(option == "4"):
    # Feature selection SVD
    svd(selected_number,feature_o)
elif(option == "5"):
    # Feature selection NMF
    nmf(selected_number,feature_o)
elif(option == "6"):
    # Feature selection MDS
    mds(selected_number,feature_o)
elif(option == "7"):
    # Feature selection ICA
    X=np.array(feature_o)
    ICA(X,n_components=selected_number)
elif(option == "8"):
    # Feature selection FA
    X=np.array(feature_o)
    FA(X,n_components=selected_number)
else:
    print("Invalid method number. Please check the method table!")



