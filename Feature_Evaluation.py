# This code is used for feature evaluation methods, 
# users must have provide labels for their sequencing data
# The output file will be generated in same folder of this code
# -------------------------------------------------------------

# load packages:
import sys,os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# find the path
#Script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# read selected features
#filename = sys.argv[1]
feature = pd.read_csv(filename,index_col=0,float_precision='round_trip')
#feature_o = pd.read_csv(feature,delim_whitespace=True,index_col=0,float_precision='round_trip')

#filename = "Lasso_out.csv"
#feature = pd.read_csv(filename,index_col=0,float_precision='round_trip')
# read feature labels
#label = sys.argv[2]
label = pd.read_csv("label.txt",delim_whitespace=True,index_col=0)
#label = pd.read_csv(label,delim_whitespace=True,index_col=0)

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(feature, label.values.ravel(), test_size=0.3)

# Evaluation method
# Feature evaluation SVM
clf_svm = SVC(gamma='scale')
clf_svm= clf_svm.fit(x_train, y_train)
ac_svm = accuracy_score(y_test,clf_svm.predict(x_test))
print('SVM Accuracy is: ',ac_svm)

# Feature evaluation KNN
clf_knn = KNeighborsClassifier()
clf_knn = clf_knn.fit(x_train, y_train)
ac_knn = accuracy_score(y_test,clf_knn.predict(x_test))
print('KNN Accuracy is: ',ac_knn)

# Feature evalution RandomForest
clf_rf  = RandomForestClassifier(n_estimators=100)
clr_rf = clf_rf.fit(x_train,y_train)
ac_rf = accuracy_score(y_test,clf_rf.predict(x_test))
print('RandomForest Accuracy is: ',ac_rf)

# Feature evaluation LightGBM
lgbc=LGBMClassifier()
lgbc = lgbc.fit(x_train, y_train)
ac_lgbc = accuracy_score(y_test,lgbc.predict(x_test))
print('LightGBM Accuracy is: ',ac_lgbc)

# Feature evaluation xgboost
clf_xgb = XGBClassifier()
clf_xgb = clf_xgb.fit(x_train, y_train)
ac_xgb = accuracy_score(y_test,clf_xgb.predict(x_test))
print('XGB Accuracy is: ',ac_xgb)

# Feature evaluation AdaBoost
clf_ada = AdaBoostClassifier()
clf_ada = clf_ada.fit(x_train, y_train)
ac_ada = accuracy_score(y_test,clf_ada.predict(x_test))
print('Adaboost Accuracy is: ',ac_ada)

# Feature evaluation Bagging Classifier
clf_bag = BaggingClassifier()
clf_bag = clf_bag.fit(x_train, y_train)
ac_bag = accuracy_score(y_test,clf_bag.predict(x_test))
print('Bagging Accuracy is: ',ac_bag)

# Feature evaluation ExtraTree
clf_tree = ExtraTreesClassifier(n_estimators=10)
clf_tree = clf_tree.fit(x_train, y_train)
ac_tree = accuracy_score(y_test,clf_tree.predict(x_test))
print('ExtraTrees Accuracy is: ',ac_tree)

# Feature evaluation Gaussian Naive Bayes
gnb = GaussianNB()
gnb = gnb.fit(x_train, y_train)
ac_gnb = accuracy_score(y_test,gnb.predict(x_test))
print('Gaussian Naive Bayes Accuracy is: ',ac_gnb)

# Feature evaluaton Gradient Boosting
gbc = GradientBoostingClassifier()
gbc = gbc.fit(x_train, y_train)
ac_gbc = accuracy_score(y_test,gbc.predict(x_test))
print('Gradient Boosting Accuracy is: ',ac_gbc)

# Feature evaluation comparision
plt.rcdefaults()
fig, ax = plt.subplots()

methods = ('SVM', 'KNN', 'RandomForest', 'LightGBM','xgboost','AdaBoost','Bagging','ExtraTree', 'Gaussian Naive Bayes','Gradient Boosting')
y_pos = np.arange(len(methods))
performance = [ac_svm,ac_knn,ac_rf,ac_lgbc,ac_xgb,ac_ada,ac_bag,ac_tree,ac_gnb,ac_gbc]
ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Classification Accuracy')
ax.set_title('Feature selection methods evluation')
ax.grid(True,axis='x')
plt.savefig('EvaluationComparsion.pdf')
plt.show()

evluation = {'Methods':methods,'Accuracy':performance}
evluation = pd.DataFrame(evluation)
evluation.to_csv("EvaluationComparsion.csv",index=False)