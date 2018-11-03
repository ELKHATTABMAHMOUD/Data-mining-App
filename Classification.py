#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


#----------------------------------------  QUESTION 1 ---------------------------------------------------------

data = arff.loadarff('labor.arff')
labors = pd.DataFrame(data[0])

#Categorical columns
categorical = ['cost-of-living-adjustment','pension','education-allowance','vacation','longterm-disability-assistance','contribution-to-dental-plan','bereavement-assistance','contribution-to-health-plan']

#Numeric columns 
numeric = ['duration','wage-increase-first-year','wage-increase-second-year','wage-increase-third-year','working-hours','standby-pay','shift-differential','statutory-holidays']

#----------------------------------------  QUESTION 2 ---------------------------------------------------------
#Normalisation de donees numeriques 
labor_num = labors.select_dtypes(include='float64')
scaler = StandardScaler();
scaler.fit(labor_num) 
labor_num = scaler.transform(labor_num)

# Remplacement des valeurs nan par la moyenne des valeurs de la colonne 
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(labor_num)
labor_num = imp_mean.transform(labor_num)

#----------------------------------------  QUESTION 3 ---------------------------------------------------------

#----------------------------------------  QUESTION 4 ---------------------------------------------------------

#----------------------------------------  QUESTION 5 ---------------------------------------------------------

labor_cat = labors.select_dtypes(exclude='float64').drop('class',axis=1)
#for col in categorical:
#	print(labor_cat[col].value_counts())
#vote_one_hot = pd.get_dummies(votes)
#vote_one_hot.drop(vote_one_hot.filter(regex='_\?$',axis=1).columns,axis=1,inplace=True)












from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

lst_classif = [dummycl, gmb, dectree, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression', 'SVM']

def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))