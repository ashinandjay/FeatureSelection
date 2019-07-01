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
import numpy.matlib
from sklearn.metrics.pairwise import rbf_kernel
from numpy import linalg as LA
from scipy.sparse import *

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

#- Feature selection methods ---------------------------------------------------------------------

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
    lsvc = LinearSVC(penalty="l1",dual=False)
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
    return F

# Feature selection JMI
def jmi(X, y, **kwargs):
    if 'n_selected_features' in kwargs.keys():
        n_selected_features = kwargs['n_selected_features']
        F, J_CMI, MIfy = lcsi(X, y, function_name='JMI', n_selected_features=n_selected_features)
    else:
        F, J_CMI, MIfy = lcsi(X, y, function_name='JMI')
    return F

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
    return mrmd_end

# Feature selection ReliefF
def ReliefF_Method(X,y,n):
    X=np.array(X)
    y=np.asarray(y)
    y = y[:, 0]
    clf = ReliefF(n_features_to_select=n, n_neighbors=100)
    Reresult = clf.fit_transform(X,y)
    np.savetxt("ReliefF_out.csv", Reresult, delimiter=",")
    return None

# Feature selection trace_ratio
def trace_ratio(X, y, n_selected_features, **kwargs):
    import construct_W
    # if 'style' is not specified, use the fisher score way to built two affinity matrix
    if 'style' not in kwargs.keys():
        kwargs['style'] = 'fisher'
    # get the way to build affinity matrix, 'fisher' or 'laplacian'
    style = kwargs['style']
    n_samples, n_features = X.shape

    # if 'verbose' is not specified, do not output the value of objective function
    if 'verbose' not in kwargs:
        kwargs['verbose'] = False
    verbose = kwargs['verbose']

    if style is 'fisher':
        kwargs_within = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
        # build within class and between class laplacian matrix L_w and L_b
        W_within = construct_W.construct_W(X, **kwargs_within)
        L_within = np.eye(n_samples) - W_within
        L_tmp = np.eye(n_samples) - np.ones([n_samples, n_samples])/n_samples
        L_between = L_within - L_tmp

    if style is 'laplacian':
        kwargs_within = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
        # build within class and between class laplacian matrix L_w and L_b
        W_within = construct_W.construct_W(X, **kwargs_within)
        D_within = np.diag(np.array(W_within.sum(1))[:, 0])
        L_within = D_within - W_within
        W_between = np.dot(np.dot(D_within, np.ones([n_samples, n_samples])), D_within)/np.sum(D_within)
        D_between = np.diag(np.array(W_between.sum(1)))
        L_between = D_between - W_between

    # build X'*L_within*X and X'*L_between*X
    L_within = (np.transpose(L_within) + L_within)/2
    L_between = (np.transpose(L_between) + L_between)/2
    S_within = np.array(np.dot(np.dot(np.transpose(X), L_within), X))
    S_between = np.array(np.dot(np.dot(np.transpose(X), L_between), X))

    # reflect the within-class or local affinity relationship encoded on graph, Sw = X*Lw*X'
    S_within = (np.transpose(S_within) + S_within)/2
    # reflect the between-class or global affinity relationship encoded on graph, Sb = X*Lb*X'
    S_between = (np.transpose(S_between) + S_between)/2

    # take the absolute values of diagonal
    s_within = np.absolute(S_within.diagonal())
    s_between = np.absolute(S_between.diagonal())
    s_between[s_between == 0] = 1e-14  # this number if from authors' code

    # preprocessing
    fs_idx = np.argsort(np.divide(s_between, s_within), 0)[::-1]
    k = np.sum(s_between[0:n_selected_features])/np.sum(s_within[0:n_selected_features])
    s_within = s_within[fs_idx[0:n_selected_features]]
    s_between = s_between[fs_idx[0:n_selected_features]]

    # iterate util converge
    count = 0
    while True:
        score = np.sort(s_between-k*s_within)[::-1]
        I = np.argsort(s_between-k*s_within)[::-1]
        idx = I[0:n_selected_features]
        old_k = k
        k = np.sum(s_between[idx])/np.sum(s_within[idx])
        if verbose:
            print('obj at iter {0}: {1}'.format(count+1, k))
        count += 1
        if abs(k - old_k) < 1e-3:
            break

    # get feature index, feature-level score and subset-level score
    feature_idx = fs_idx[I]
    feature_score = score
    subset_score = k
    return feature_idx, feature_score, subset_score

# Feature selection gini index
def gini_index(X, y):
    n_samples, n_features = X.shape
    # initialize gini_index for all features to be 0.5
    gini = np.ones(n_features) * 0.5
    # For i-th feature we define fi = x[:,i] ,v include all unique values in fi
    for i in range(n_features):
        v = np.unique(X[:, i])
        for j in range(len(v)):
            # left_y contains labels of instances whose i-th feature value is less than or equal to v[j]
            left_y = y[X[:, i] <= v[j]]
            # right_y contains labels of instances whose i-th feature value is larger than v[j]
            right_y = y[X[:, i] > v[j]]
            # gini_left is sum of square of probability of occurrence of v[i] in left_y
            # gini_right is sum of square of probability of occurrence of v[i] in right_y
            gini_left = 0
            gini_right = 0
            for k in range(np.min(y), np.max(y)+1):
                if len(left_y) != 0:
                    # t1_left is probability of occurrence of k in left_y
                    t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                    t2_left = np.power(t1_left, 2)
                    gini_left += t2_left
                if len(right_y) != 0:
                    # t1_right is probability of occurrence of k in left_y
                    t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                    t2_right = np.power(t1_right, 2)
                    gini_right += t2_right
            gini_left = 1 - gini_left
            gini_right = 1 - gini_right
            # weighted average of len(left_y) and len(right_y)
            t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)
            # compute the gini_index for the i-th feature
            value = np.true_divide(t1_gini, len(y))
            if value < gini[i]:
                gini[i] = value
    return gini

def feature_ranking(W):
    idx = np.argsort(W)
    return idx

# Feature selection SPEC
def spec(X, **kwargs):
    if 'style' not in kwargs:
        kwargs['style'] = 0
    if 'W' not in kwargs:
        kwargs['W'] = rbf_kernel(X, gamma=1)

    style = kwargs['style']
    W = kwargs['W']
    if type(W) is numpy.ndarray:
        W = csc_matrix(W)
    n_samples, n_features = X.shape
    # build the degree matrix
    X_sum = np.array(W.sum(axis=1))
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        D[i, i] = X_sum[i]
    # build the laplacian matrix
    L = D - W
    d1 = np.power(np.array(W.sum(axis=1)), -0.5)
    d1[np.isinf(d1)] = 0
    d2 = np.power(np.array(W.sum(axis=1)), 0.5)
    v = np.dot(np.diag(d2[:, 0]), np.ones(n_samples))
    v = v/LA.norm(v)
    # build the normalized laplacian matrix
    L_hat = (np.matlib.repmat(d1, 1, n_samples)) * np.array(L) * np.matlib.repmat(np.transpose(d1), n_samples, 1)
    # calculate and construct spectral information
    s, U = np.linalg.eigh(L_hat)
    s = np.flipud(s)
    U = np.fliplr(U)
    # begin to select features
    w_fea = np.ones(n_features)*1000
    for i in range(n_features):
        f = X[:, i]
        F_hat = np.dot(np.diag(d2[:, 0]), f)
        l = LA.norm(F_hat)
        if l < 100*np.spacing(1):
            w_fea[i] = 1000
            continue
        else:
            F_hat = F_hat/l
        a = np.array(np.dot(np.transpose(F_hat), U))
        a = np.multiply(a, a)
        a = np.transpose(a)

        # use f'Lf formulation
        if style == -1:
            w_fea[i] = np.sum(a * s)
        # using all eigenvalues except the 1st
        elif style == 0:
            a1 = a[0:n_samples-1]
            w_fea[i] = np.sum(a1 * s[0:n_samples-1])/(1-np.power(np.dot(np.transpose(F_hat), v), 2))
        # use first k except the 1st
        else:
            a1 = a[n_samples-style:n_samples-1]
            w_fea[i] = np.sum(a1 * (2-s[n_samples-style: n_samples-1]))

    if style != -1 and style != 0:
        w_fea[w_fea == 1000] = -1000
    return w_fea

def feature_ranking2(score, **kwargs):
    if 'style' not in kwargs:
        kwargs['style'] = 0
    style = kwargs['style']

    # if style = -1 or 0, ranking features in descending order, the higher the score, the more important the feature is
    if style == -1 or style == 0:
        idx = np.argsort(score, 0)
        return idx[::-1]
    # if style != -1 and 0, ranking features in ascending order, the lower the score, the more important the feature is
    elif style != -1 and style != 0:
        idx = np.argsort(score, 0)
        return idx

# Feature selection Fisher Score
import construct_W
def fisher_score(X, y):
    # Construct weight matrix W in a fisherScore way
    kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
    W = construct_W.construct_W(X, **kwargs)
    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    lap_score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]
    # compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
    score = 1.0/lap_score - 1
    return np.transpose(score)

def feature_ranking3(score):
    idx = np.argsort(score, 0)
    return idx[::-1]

# Feature selection t_score
def t_score(X, y):
    n_samples, n_features = X.shape
    F = np.zeros(n_features)
    c = np.unique(y)
    if len(c) == 2:
        for i in range(n_features):
            f = X[:, i]
            # class0 contains instances belonging to the first class
            # class1 contains instances belonging to the second class
            class0 = f[y == c[0]]
            class1 = f[y == c[1]]
            mean0 = np.mean(class0)
            mean1 = np.mean(class1)
            std0 = np.std(class0)
            std1 = np.std(class1)
            n0 = len(class0)
            n1 = len(class1)
            t = mean0 - mean1
            t0 = np.true_divide(std0**2, n0)
            t1 = np.true_divide(std1**2, n1)
            F[i] = np.true_divide(t, (t0 + t1)**0.5)
    else:
        print('y should be guaranteed to a binary class vector')
        exit(0)
    return np.abs(F)

def feature_ranking4(F):
    idx = np.argsort(F)
    return idx[::-1]

# Feature selection IG
def calProb(array):
	myProb = {}
	myClass = set(array)
	for i in myClass:
		myProb[i] = array.count(i) / len(array)
	return myProb

def jointProb(newArray, labels):
	myJointProb = {}
	for i in range(len(labels)):
		myJointProb[str(newArray[i]) + '-' + str(labels[i])] = myJointProb.get(str(newArray[i]) + '-' + str(labels[i]), 0) + 1

	for key in myJointProb:
		myJointProb[key] = myJointProb[key] / len(labels)
	return myJointProb

def IG(encodings,labelfile,k):
    encoding=np.array(encodings)
    sample=encoding[:,0]
    data=encoding[:,1:]
    shape=data.shape
    data = np.reshape(data, shape[0] * shape[1])
    data = np.reshape([float(i) for i in data], shape)
    samples=[i for i in sample]
    file = open(labelfile,'r')
    file.readline()
    for line in file.readlines():
        records=line[:]
    myDict = {}
    try:
	    for i in records:
		     array = i.rstrip().split() if i.strip() != '' else None
		     myDict[array[0]] = int(array[1])
    except IndexError as e:
        print(e)
    labels = []
    for i in samples:
	     labels.append(myDict.get(i, 0))

    dataShape = data.shape
    features=range(dataShape[1])

    if dataShape[0] != len(labels):
	    print('Error: inconsistent data shape with sample number.')
    probY = calProb(labels)
    myFea = {}
    binBox = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(len(features)):
	     array = data[:, i]
	     newArray = list(pd.cut(array, len(binBox), labels= binBox))
	     probX = calProb(newArray)
	     probXY = jointProb(newArray, labels)
	     HX = -1 * sum([p * math.log(p, 2) for p in probX.values()])
	     HXY = 0
	     for y in probY.keys():
		     for x in probX.keys():
			     if str(x) + '-' + str(y) in probXY:
				     HXY = HXY + (probXY[str(x) + '-' + str(y)] * math.log(probXY[str(x) + '-' + str(y)] / probY[y], 2))
	     myFea[features[i]] = HX + HXY
    res=[]
    for key in sorted(myFea.items(), key=lambda item:item[1], reverse=True):
	    res.append([key[0], '{0:.3f}'.format(myFea[key[0]])]) 
    res=np.array(res)
    importance=res[:,0]
    feature_=np.array([float(i) for i in importance])
    mask=feature_[:k].astype(int)
    new_data=data[:,mask]
    return new_data

#- main function ----------------------------------------------------------------------

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
    cor_support, cor_feature = cor_selector(selected_number,feature_o, label.values.ravel())
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
    MRMRresult=mrmr(X,y,n_selected_features=selected_number)
    MRMRresult = feature_o[feature_o.columns[MRMRresult]]
    MRMRresult.to_csv("MRMR_out.csv")
elif(option == "12"):
    # Feature selection JMI
    X=np.array(feature_o)
    y=np.asarray(label)
    y = y[:, 0]
    JMIresult=jmi(X,y,n_selected_features=selected_number)
    JMIresult = feature_o[feature_o.columns[JMIresult]]
    JMIresult.to_csv("JMI_out.csv")
elif(option == "13"):
    # Feature selection MRMD
    X=np.array(feature_o)
    y=np.array(label)
    y = y[:, 0]
    MRMDresult=mrmd(X,y,n_selected_features=selected_number)
    MRMDresult = feature_o[feature_o.columns[MRMDresult]]
    MRMDresult.to_csv("MRMD_out.csv")
elif(option == "14"):
    # Feature selection ReliefF
    ReliefF_Method(feature_o,label,selected_number)
elif(option == "15"):
    # Feature selection trace_ratio
    X=np.array(feature_o)
    y=np.array(label)
    y=y[:,0]
    idx, feature_score, subset_score = trace_ratio(X, y, selected_number, style='fisher')
    traceresult = feature_o[feature_o.columns[idx]]
    traceresult.to_csv("TRACE_RATIO_out.csv")
elif(option == "16"):
    # Feature selection gini_index
    X=np.array(feature_o)
    y=np.asarray(label)
    y = y[:, 0]
    score = gini_index(X, y)
    # rank features in descending order according to score
    idx =feature_ranking(score)
    giniresult = feature_o[feature_o.columns[idx[0:selected_number]]]
    giniresult.to_csv("GINI_INDEX_out.csv")
elif(option == "17"):
    # Feature selection SPEC
    X=np.array(feature_o)
    y=np.asarray(label)
    y = y[:, 0] 
    # specify the second ranking function which uses all except the 1st eigenvalue
    kwargs = {'style': 0}
    # obtain the scores of features
    score = spec(X, **kwargs)
    # sort the feature scores in an descending order according to the feature scores
    idx = feature_ranking2(score, **kwargs)
    specresult = feature_o[feature_o.columns[idx[0:selected_number]]]
    specresult.to_csv("SPEC_out.csv")
elif(option == "18"):
    # Feature selection Fisher_score
    X=np.array(feature_o)
    y=np.asarray(label)
    y = y[:, 0]
    score = fisher_score(X, y)
    # rank features in descending order according to score
    idx =feature_ranking3(score)
    fisherresult = feature_o[feature_o.columns[idx[0:selected_number]]]
    fisherresult.to_csv("Fisher_Score_out.csv")
elif(option == "19"):
    # Feature selection t_score
    X=np.array(feature_o)
    y=np.asarray(label)
    y = y[:, 0]
    score = t_score(X, y)
    idx =feature_ranking4(score)
    tscoreresult = feature_o[feature_o.columns[idx[0:selected_number]]]
    tscoreresult.to_csv("T_Score_out.csv")
elif(option == "20"):
    # Feature selection IG
    IGresult=IG(feature_o,label,selected_number)
    IGresult.to_csv("IG_out.csv")
else:
    print("Invalid method number. Please check the method table!")
    
