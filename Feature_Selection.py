# This code is used for feature selection methods, 
# users must have provide labels for their sequencing data
# The output file will be generated in same folder of this code
# -------------------------------------------------------------

# Load packages:
import sys,os
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import math
import scipy.spatial as ss
from scipy.special import digamma
from math import log
import numpy.random as nr
from skrebate import ReliefF

# find the path
Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# choose the method
option = sys.argv[1]

# read number of feature to select
selected_number = sys.argv[2]

# read feature vectors
feature = sys.argv[3]
feature_o = pd.read_csv(feature,delim_whitespace=True,index_col=0,float_precision='round_trip')

# read feature labels
label = sys.argv[4]
label = pd.read_csv(label,delim_whitespace=True,index_col=0)

# Feature selection methods:

# Feature selection Lasso    
def Lasso_selection(k,X,y):
    clf = LassoCV(cv=3)
    lassoresult = RFE(clf,k).fit(X,y.values.ravel()).get_support(indices=True)
    lassoresult = X[X.columns[lassoresult]]
    lassoresult.to_csv("Lasso_out.csv")
    return None

# Feature selection Elastic-Net
def Elastic_selection(k,X,y):
    elasticnet = ElasticNet()
    elasticresult = RFE(elasticnet,k).fit(X,y.values.ravel()).get_support(indices=True)
    elasticresult = X[X.columns[elasticresult]]
    elasticresult.to_csv("ElasticNet_out.csv")
    return None

# Feature selection L1-SVM
def L1SVM_selection(k,X,y):
    lsvc = LinearSVC(penatly="l1",dual=False)
    lsvcresult = RFE(lsvc,k).fit(X,y.values.ravel()).get_support(indices=True)
    lsvcresult = X[X.columns[lsvcresult]]
    lsvcresult.to_csv("L1SVM_out.csv")
    return None

# Feature selection CHI2
def Chi2_selection(k,X,y):
    chi2result = SelectKBest(chi2, k).fit(X, y).get_support(indices=True)
    chi2result = X[X.columns[chi2result]]
    chi2result.to_csv("CHI2_out.csv")
    return None

# Feature selection Pearson Correlation
def cor_selector(k, X, y):
    feature_name=X.columns.tolist()
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-k:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

# Feature selection ExtraTree
def ExtraTree_selection(k,X,y):
    extraT = ExtraTreesClassifier(n_estimators=10)
    extraresult = RFE(extraT,k).fit(X,y.values.ravel()).get_support(indices=True)
    extraresult = X[X.columns[extraresult]]
    extraresult.to_csv("ExtraTree_out.csv")
    return None

# Feature selection XGBoost
def XGB_selection(k,X,y):
    xgb = XGBClassifier()
    xgbresult = RFE(xgb,k).fit(X,y.values.ravel()).get_support(indices=True)
    xgbresult = X[X.columns[xgbresult]]
    xgbresult.to_csv("XGBoost_out.csv")
    return None

# Feature selection SVM-RFE
def SVM_RFE_selection(k,X,y):
    svc = SVC(kernel="linear")
    svmresult = RFE(svc,k).fit(X,y.values.ravel()).get_support(indices=True)
    svmresult = X[X.columns[svmresult]]
    svmresult.to_csv("SVM-RFE_out.csv")
    return None

# Feature selection LOG-RFE
def LOG_RFE_selection(k,X,y):
    log = LogisticRegression(solver='liblinear')
    logresult = RFE(log,k).fit(X,y.values.ravel()).get_support(indices=True)
    logresult = X[X.columns[logresult]]
    logresult.to_csv("LOG-FRE_out.csv")
    return None

# Feature selection Mutual Information
def Mutual_Info_selection(k,X,y):
    model_mutual = SelectKBest(mutual_info_classif, k)
    mutual = model_mutual.fit(X,y.values.ravel()).get_support(indices=True)
    mutualresult = X[X.columns[mutual]]
    mutualresult.to_csv("MutualInfo_out.csv")
    return None

# Feature selection MRMR
def entropy(x, k=3, base=2):    
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    d = len(x[0])
    N = len(x)
    intens = 1e-10  # small noise to break degeneracy, see doc.
    x = [list(p + intens * nr.rand(len(x[0]))) for p in x]
    tree = ss.cKDTree(x)
    nn = [tree.query(point, k+1, p=float('inf'))[0][k] for point in x]
    const = digamma(N)-digamma(k) + d*log(2)
    return (const + d*np.mean(map(log, nn)))/log(base)

def entropyd(sx, base=2):  # Discrete estimators
    return entropyfromprobs(hist(sx), base=base)

def midd(x, y):
    return -entropyd(list(zip(x, y)))+entropyd(x)+entropyd(y)

def cmidd(x, y, z):
    return entropyd(list(zip(y, z)))+entropyd(list(zip(x, z)))-entropyd(list(zip(x, y, z)))-entropyd(z)

def hist(sx):
    # Histogram from list of samples
    d = dict()
    #d=list()
    for s in sx:
        d[s] = d.get(s, 0) + 1
    return map(lambda z: float(z)/len(sx), d.values())

def entropyfromprobs(probs, base=2):
    # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
    return -sum(map(elog, probs))/log(base)

def elog(x):
    # for entropy, 0 log 0 = 0. but we get an error for putting log 0
    if x <= 0. or x >= 1.:
        return 0
    else:
        return x*log(x)

def micd(x, y, k=3, base=2, warning=True): # Mixed estimators
    overallentropy = entropy(x, k, base)
    n = len(y)
    word_dict = dict()
    for sample in y:
        word_dict[sample] = word_dict.get(sample, 0) + 1./n
    yvals = list(set(word_dict.keys()))
    mi = overallentropy
    for yval in yvals:
        xgiveny = [x[i] for i in range(n) if y[i] == yval]
        if k <= len(xgiveny) - 1:
            mi -= word_dict[yval]*entropy(xgiveny, k, base)
        else:
            if warning:
                print("Warning, after conditioning, on y={0} insufficient data. Assuming maximal entropy in this case.".format(yval))
            mi -= word_dict[yval]*overallentropy
    return mi  # units already applied
	
def zip2(*args):
    return [sum(sublist, []) for sublist in zip(*args)]

def lcsi(X, y, **kwargs):    
    n_samples, n_features = X.shape
    # index of selected features, initialized to be empty
    F = []
    # Objective function value for selected features
    J_CMI = []
    # Mutual information between feature and response
    MIfy = []
    # indicate whether the user specifies the number of features
    is_n_selected_features_specified = False
    # initialize the parameters
    if 'beta' in kwargs.keys():
        beta = kwargs['beta']
    if 'gamma' in kwargs.keys():
        gamma = kwargs['gamma']
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        is_n_selected_features_specified = True
    # select the feature whose j_cmi is the largest
    # t1 stores I(f;y) for each feature f
    t1 = np.zeros(n_features)
    # t2 stores sum_j(I(fj;f)) for each feature f
    t2 = np.zeros(n_features)
    # t3 stores sum_j(I(fj;f|y)) for each feature f
    t3 = np.zeros(n_features)
    for i in range(n_features):
        f = X[:, i]
        t1[i] = midd(f, y)
    # make sure that j_cmi is positive at the very beginning
    j_cmi = 1
    while True:
        if len(F) == 0:
            # select the feature whose mutual information is the largest
            idx = np.argmax(t1)
            F.append(idx)
            J_CMI.append(t1[idx])
            MIfy.append(t1[idx])
            f_select = X[:, idx]
        if is_n_selected_features_specified:
            if len(F) == n_selected_features:
                break
        else:
            if j_cmi < 0:
                break
        # we assign an extreme small value to j_cmi to ensure it is smaller than all possible values of j_cmi
        j_cmi = -1E30
        if 'function_name' in kwargs.keys():
            if kwargs['function_name'] == 'MRMR':
                beta = 1.0 / len(F)
            elif kwargs['function_name'] == 'JMI':
                beta = 1.0 / len(F)
                gamma = 1.0 / len(F)
        for i in range(n_features):
            if i not in F:
                f = X[:, i]
                t2[i] += midd(f_select, f)
                t3[i] += cmidd(f_select, f, y)
                # calculate j_cmi for feature i (not in F)
                t = t1[i] - beta*t2[i] + gamma*t3[i]
                # record the largest j_cmi and the corresponding feature index
                if t > j_cmi:
                    j_cmi = t
                    idx = i
        F.append(idx)
        J_CMI.append(j_cmi)
        MIfy.append(t1[idx])
        f_select = X[:, idx]
    return np.array(F), np.array(J_CMI), np.array(MIfy)

def mrmr(X, y, **kwargs):    
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = lcsi(X, y, gamma=0, function_name='MRMR', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = lcsi(X, y, gamma=0, function_name='MRMR')
    X_new=X[:,F]
    return X_new

# Feature selection JMI
def jmi(X, y, **kwargs):
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = lcsi(X, y, function_name='JMI', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = lcsi(X, y, function_name='JMI')
    X_new=X[:,F]
    return X_new

# Feature selection MRMD
def calcE(X,coli,colj):
    sum1 = np.sum((X[:,coli]-X[:,colj])**2)  
    return math.sqrt(sum1)

def Euclidean(X,n):
    Euclideandata=np.zeros([n,n])    
    for i in range(n):
        for j in range(n):
            Euclideandata[i,j]=calcE(X,i,j)
            Euclideandata[j,i]=Euclideandata[i,j]
    Euclidean_distance=[]

    for i in range(n):
        sum1 = np.sum(Euclideandata[i,:])
        Euclidean_distance.append(sum1/n)
    return Euclidean_distance

def varience(data,avg1,col1,avg2,col2):
    return np.average((data[:,col1]-avg1)*(data[:,col2]-avg2))

def Person(X,y,n):
    feaNum=n
    #label_num=len(y[0,:])
    label_num=1
    PersonData=np.zeros([n])
    for i in range(feaNum):
        for j in range(feaNum,feaNum+label_num):
            #print('. ', end='')
            average1 = np.average(X[:,i])
            average2 = np.average(y)
            yn=(X.shape)[0]
            y=y.reshape((yn,1))
            dataset = np.concatenate((X,y),axis=1)
            numerator = varience(dataset, average1, i, average2, j);
            denominator = math.sqrt(
                varience(dataset, average1, i, average1, i) * varience(dataset, average2, j, average2, j));
            if (abs(denominator) < (1E-10)):
                PersonData[i]=0
            else:
                PersonData[i]=abs(numerator/denominator)
    return list(PersonData)

def mrmd(X,y,n_selected_features=10):
    n=X.shape[1]
    e=Euclidean(X,n)
    p = Person(X,y,n)
    mrmrValue=[]
    for i,j in zip(p,e):
        mrmrValue.append(i+j)
    mrmr_max=max(mrmrValue)
    features_name=np.array(range(n))
    mrmrValue = [x / mrmr_max for x in mrmrValue]
    mrmrValue = [(i,j) for i,j in zip(features_name,mrmrValue)]   
    mrmd_order=sorted(mrmrValue,key=lambda x:x[1],reverse=True)  
    mrmd_order =[int(x[0]) for x in mrmd_order]
    mrmd_end=mrmd_order[:n_selected_features]
    X_new=X[:,mrmd_end]
    return X_new

# Feature selection ReliefF
def ReliefF(X,y,n):
    X=np.array(X)
    y=np.asarray(y)
    y = y[:, 0]
    clf = ReliefF(n_features_to_select=3, n_neighbors=100)
    Reresult = clf.fit_transform(X,y)
    Reresult.to_csv("ReliefF_out.csv")
    return None

# main function

if(option == "1"):
    # Feature selection Lasso
    Lasso_selection(selected_number,feature_o,label)
elif(option == "2"):
    # Feature selection Elastic-Net
    Elastic_selection(selected_number,feature_o,label)
elif(option == "3"):
    # Feature selection L1-SVM
    L1SVM_selection(selected_number,feature_o,label)
elif(option == "4"):
    # Feature selection CHI2
    Chi2_selection(selected_number,feature_o,label)
elif(option == "5"):
    # Feature selection Pearson Correlation
    cor_support, cor_feature = cor_selector(feature_o, label.values.ravel())
    PCresult = feature_o[cor_feature]
    PCresult.to_csv("PearsonCorrelation_out.csv")
elif(option == "6"):
    # Feature selection ExtraTree   
    ExtraTree_selection(selected_number,feature_o,label)
elif(option == "7"):
    # Feature selection XGBoost
    XGB_selection(selected_number,feature_o,label)
elif(option == "8"):
    # Feature selection SVM-RFE 
    SVM_RFE_selection(selected_number,feature_o,label)
elif(option == "9"):
    # Feature selection LOG-RFE
    LOG_RFE_selection(selected_number,feature_o,label)
elif(option == "10"):
    # Feature selection Mutual Info
    Mutual_Info_selection(selected_number,feature_o,label)
elif(option == "11"):
    # Feature selection MRMR
    X=np.array(feature_o)
    y=np.asarray(label)
    y = y[:, 0]
    MRMRresult=mrmr(X,y,selected_number)
    MRMRresult.to_csv("MRMR_out.csv")
elif(option == "12"):
    # Feature selection JMI
    X=np.array(feature_o)
    y=np.asarray(label)
    y = y[:, 0]
    JMIresult=mrmd(X,y,selected_number)
    JMIresult.to_csv("JMI_out.csv")
elif(option == "13"):
    # Feature selection MRMD
    X=np.array(feature_o)
    y=np.array(label)
    MRMDresult=mrmd(X,y,selected_number)
    MRMDresult.to_csv("MRMD_out.csv")
elif(option == "14"):
    # Feature selection ReliefF
    ReliefF(feature_o,label,selected_number)
elif(option == "15"):
    # Feature selection trace_ratio
elif(option == "16"):
    # Feature selection gini_index
elif(option == "17"):
    # Feature selection t_score
elif(option == "18"):
    # Feature selection SPEC
elif(option == "19"):
    # Feature selection Fisher_score
elif(option == "20"):
    # Feature selection LFDA
else:
    print("Invalid method number. Please check the method table!")
    
