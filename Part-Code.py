#import the Cleveland heart patient data file using pandas, creating a header row

import pandas as pnd #importing pandas data analysis toolkit
header_row = ['age','sex','pain','BP','chol','fbs','ecg','maxhr','eiang','eist','slope','vessels','thal','diagnosis']
heart = pnd.read_csv('processed.cleveland.data', names=header_row)

print("Unprocessed Cleveland Dataset")
print("************************************************************************")
print(heart.loc[:,'age':'diagnosis'])
#print("************************************************************************")

#filter to only those diagnosed with heart disease

#print(heart['diagnosis'].value_counts())
has_hd_check = heart['diagnosis'] > 0
has_hd_patients = heart[has_hd_check]
#print(has_hd_patients)
#add this column to dataframe
heart['diag_bool'] = has_hd_check
#also add integer version
heart['diag_int'] = has_hd_check.astype(int)




#need to turn thal and vessels slope into floats (currently objects because of ? values)
import numpy as np
heart['vessels'] = heart['vessels'].apply(lambda vessels: 0.0 if vessels == "?" else vessels)
#print(heart['vessels'])
heart['vessels'] = heart['vessels'].astype(float)
heart['thal'] = heart['thal'].apply(lambda thal: 0.0 if thal == "?" else thal)
heart['thal'] = heart['thal'].astype(float)

print("Processed Cleveland Dataset")
print("************************************************************************")
print(heart.loc[:,'age':'diagnosis'])
print("************************************************************************")

#import and modify VA dataset for testing
heart_va = pnd.read_csv('processed.va.data', names=header_row)
print("Unprocessed VA Dataset")
print("************************************************************************")
print(heart_va.loc[:,'age':'diagnosis'])
print("************************************************************************")

has_hd_check = heart_va['diagnosis'] > 0
heart_va['diag_int'] = has_hd_check.astype(int)
heart_va = heart_va.replace(to_replace = '?',value = 0.0)

print("Processed VA Dataset")
print("************************************************************************")
print(heart_va.loc[:,'age':'diagnosis'])
print("************************************************************************")

#import and modify hungarian dataset for testing
heart_hu = pnd.read_csv('processed.hungarian.data', names=header_row)
print("Unprocessed Hungarian Dataset")
print("************************************************************************")
print(heart_hu.loc[:,'age':'diagnosis'])
print("************************************************************************")

has_hd_check = heart_hu['diagnosis'] > 0
heart_hu['diag_int'] = has_hd_check.astype(int)
heart_hu = heart_hu.replace(to_replace = '?',value = 0.0)

print("Processed Hungarian Dataset")
print("************************************************************************")
print(heart_hu.loc[:,'age':'diagnosis'])
print("************************************************************************")



#classification with scikit-learn SVM
from time import time
from sklearn import datasets

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Splitting data for training and testing

x_train1, x_test1, y_train1, y_test1 = train_test_split(heart.loc[:,'age':'thal'], heart.loc[:,'diag_int'], test_size=0.30, random_state=42)
x_train2, x_test2, y_train2, y_test2 = train_test_split(heart_va.loc[:,'age':'thal'], heart_va.loc[:,'diag_int'], test_size=0.30, random_state=42)
x_train3, x_test3, y_train3, y_test3 = train_test_split(heart_hu.loc[:,'age':'thal'], heart_hu.loc[:,'diag_int'], test_size=0.30, random_state=42)

# Combining the dataset for Cleveland, VA and Hungarian Dataset

x_train=np.concatenate(   [x_train1,
            x_train2,
            x_train3],axis=0)

y_train=np.concatenate(   [y_train1,
            y_train2,
            y_train3],axis=0)


x_test=np.concatenate(    [x_test1,
            x_test2,
            x_test3],axis=0)

y_test=np.concatenate(    [y_test1,
            y_test2,
            y_test3],axis=0)


# Classification using SVM
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    
clf.fit( x_train,y_train)   # Fit the classification boundary
#print('************************************************************************')

#print("Best estimator found by grid search:")
#print(clf.best_estimator_)
#print('************************************************************************')


t0 = time()
y_pred = clf.predict(x_test)
t1= time()-t0
print('')
print("Time for SVM Classification:")
print(t1)
y_pred = clf.predict(x_test)
print(y_pred)
print(y_test)
print('Results for SVM Classification:')
target_names=['With Heart Disease','Without Heart Disease']
print(classification_report(y_test, y_pred,target_names=target_names))
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
