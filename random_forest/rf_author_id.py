#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 4 Mini Project - Choose your own algorithm.

    Use a Random Forest Classifier to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from prep_terrain_data import makeTerrainData
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, labels_train, features_test, labels_test = makeTerrainData(2000)


def submitAccuracies():
  return {"acc":round(acc,3)}

#########################################################
### your code goes here ###
#########################################################
clf = RandomForestClassifier(n_estimators=10, max_depth=2, min_samples_split=50,
                             random_state=0)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

acc = clf.score(features_test, labels_test)
print submitAccuracies()

# number of features
print "Number of features: {}".format(len(features_train))