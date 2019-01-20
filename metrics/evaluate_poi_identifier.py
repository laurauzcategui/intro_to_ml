#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,  test_size=0.3, random_state=42)


### your code goes here 
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(pred, labels_test)

print("second accuracy: {}".format(acc))

# How many POIs are predicted for the test set for your POI identifier?
print("POIs predicted: {}".format(sum(labels_test)))

# How many in total are in the test set? 
print("Test set total: {}".format(len(labels_test)))

# If your identifier predicted 0. (not POI)
# for everyone in the test set, what would its accuracy be?
print("Accuracy not POI:{}".format(1 - (sum(labels_test)/len(labels_test))))

# Look at the predictions of your model and 
# # compare them to the true test labels. 
# Do you get any true positives? (In this case, 
# we define a true positive as a case where both the actual label and the predicted label are 1)
cnt = 0
for idx, e in enumerate(pred):
    if e == 1.0 and labels_test[idx] == 1.0:
        cnt += 1
print("True Positives: {}".format(cnt))

# Whats the precision? 
from sklearn.metrics import precision_score, recall_score
print("Precision Score: {}".format(precision_score(labels_test,pred)))

# Whats the recall?  
print("Recall Score: {}".format(recall_score(labels_test,pred)))

# pred [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
# true [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

# How Many true positives? 6 

# How many true negatives? 9

# False positives? 3

# False Negatives? 2

# Precision? tp / tp + fp = 6 / (6 + 3) --> .66

# Recall ? tp / tp + fn = 6 / ( 6 + 2 ) --> .75