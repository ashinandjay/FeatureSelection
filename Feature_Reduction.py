# This code is used for feature Reduction methods, 
# The output file will be generated in same folder of this code
# -------------------------------------------------------------

# load packages:
import sys,os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

# find the path
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# choose the method
option = sys.argv[1]

# read number of clustering
selected_number = sys.argv[2]

# read feature vectors
feature = sys.argv[3]
feature_o = pd.read_csv(feature,delim_whitespace=True,index_col=0,float_precision='round_trip')

# Feature selection Kmeans
def kmeansselection(k,X):
    X=np.array(X)
    kmeansresult=KMeans(n_clusters = k).fit_transform(X)
    np.savetxt("KMeans_out.csv", kmeansresult, delimiter=",")    
    return None   

# Feature selection tsne
def tsneselection(k,X):
    X=np.array(X)
    Tsneresult =TSNE(n_components=k).fit_transform(X)
    np.savetxt("TSNE_out.csv", Tsneresult, delimiter=",")    
    return None 

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

# Feature selection Agglomerate features
def Aggselection(k,X):
    X=np.array(X)
    Aggresult =FeatureAgglomeration(n_clusters=k).fit_transform(X)
    np.savetxt("Agglomeration_out.csv", Aggresult, delimiter=",")    
    return None 

# Feature selection Gaussian random projection
def Gaussianselection(k,X):
    X=np.array(X)
    gaussiansresult=GaussianRandomProjection(n_components=k).fit_transform(X)
    np.savetxt("Gaussian_out.csv", gaussiansresult, delimiter=",")
    return None

# Feature selection Sparse random projection
def Sparseselection(k,X):
    X=np.array(X)
    sparsesresult=SparseRandomProjection(n_components=k).fit_transform(X)
    np.savetxt("Sparse_out.csv", sparsesresult, delimiter=",")
    return None

# main code

if(option == "1"):
    # Feature selection Kmeans
    kmeansselection(selected_number,feature_o)
elif(option == "2"):
    # Feature selection T-SNE
    tsneselection(selected_number,feature_o)
elif(option == "3")
    # Feature selection PCA
    pcaselection(selected_number,feature_o)
elif(option == "4"):
    # Feature selection KernelPCA
    kernelPCA(selected_number,feature_o)
elif(option == "5"):
    # Feature selection LLE
    lle(selected_number,feature_o)
elif(option == "6"):
    # Feature selection SVD
    svd(selected_number,feature_o)
elif(option == "7"):
    # Feature selection NMF
    nmf(selected_number,feature_o)
elif(option == "8"):
    # Feature selection MDS
    mds(selected_number,feature_o)
elif(option == "9"):
    # Feature selection ICA
    X=np.array(feature_o)
    ICA(X,n_components=selected_number)
elif(option == "10"):
    # Feature selection FA
    X=np.array(feature_o)
    FA(X,n_components=selected_number)
elif(option == "11"):
    # Feature selection Agglomerate feature
    Aggselection(selected_number,feature_o)
elif(option == "12"):
    # Feature selection Gaussian Random Projection
    Gaussianselection(selected_number,feature_o)
elif(option == "13"):
    # Feature selection Sparse Random Projection
    Sparseselection(selected_number,feature_o)
else:
    print("Invalid method number. Please check the method table!")



