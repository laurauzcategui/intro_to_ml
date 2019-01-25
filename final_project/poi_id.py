#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','long_term_incentive','loan_advances',
                 'director_fees','from_poi_to_this_person',
                 'from_this_person_to_poi', 'shared_receipt_with_poi', 'total_payments'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Exploring the data set 
''' Let us explore the dataset to be more clear on how many data 
points do we have, also the amount of features and where we might want to apply 
cleaning of outliers''' 
# Total Number of points
print("Total number of points:{}".format(len(data_dict)))
# Exploring the data_dict 
for item in data_dict:
    number_features = len(data_dict[item])
    print("Number of features:{}".format(number_features))
    break
# Get features with missing values
missing_values_features = {}
for item in data_dict:
    for feature in data_dict[item]:
        if data_dict[item][feature] == 'NaN':
            if missing_values_features.get(feature) is None:
                missing_values_features[feature] = 1
            else: 
                missing_values_features[feature] += 1

print("List of missing values by feature: {}".format(missing_values_features))

### Task 2: Remove outliers
### For checking outliers we will take some features from the initial selected features
### and we will evaluate through matplotlib 
#data outliers selection 
outliers_salary = ['salary', 'bonus']
outliers_incentives_loan = ['long_term_incentive', 'loan_advances']

def print_outliers(features):
    data = featureFormat(data_dict, features, sort_keys = True)
    for point in data:
        salary = point[0]
        bonus = point[1]
        matplotlib.pyplot.scatter( salary, bonus )
print_outliers(outliers_salary)
matplotlib.pyplot.title("Salary vs Bonus")
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
print_outliers(outliers_incentives_loan)
matplotlib.pyplot.title("Long Termin Incentive vs Load Advance")
matplotlib.pyplot.xlabel("long_term_incentive")
matplotlib.pyplot.ylabel("loan_advances")
matplotlib.pyplot.show()

print("******** Possible Outliers and POIs ( Salary & Bonus ) **********")
for key, value in data_dict.items():
    if value['bonus'] == 'NaN' or value['salary'] == 'NaN':
        continue
    
    if value['bonus'] >= 5000000 and value['salary'] >= 1000000:
        print key, value['bonus'], value['salary']
print("******** End Possible Outliers and POIs **********") 

print("******** Possible Outliers and POIs - ( Long Term Incentive & loan Advance) **********")
for key, value in data_dict.items():
    if value['long_term_incentive'] == 'NaN' or value['loan_advances'] == 'NaN':
        continue
    
    if value['long_term_incentive'] >= 1000000  and value['loan_advances'] >= 1000000:
        print key, value['long_term_incentive'], value['loan_advances']
print("******** End Possible Outliers and POIs - ( Long Term Incentive & loan Advance) **********")

# REMOVING THE OUTLIER
data_dict.pop( "TOTAL", 0 ) 

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
import os
from poi_email_addresses import poiEmails
EMAILS="./emails_by_address/"

POI_EMAILS = poiEmails()

def process_emails():
    ''' extract filename of each file in emails_by_address folder
        and pass it to process_email method
    '''
    for _, _, files in os.walk(EMAILS):  
        for filename in files:
            process_email(filename)

def process_email(email):
    ''' Will extract from title and verify if it was a sender or recipient
        It will open the file and count the amount of deleted emails by that person
    '''
    # get the email without the domain
    from_to_email = email.split("@")[0]
    domain = email.split('@')[1].split('.txt')[0]
    # build full path to file to explore 
    file_path = "{}{}".format(EMAILS,email)
    
    if from_to_email.startswith( 'from_' ):
        sender = from_to_email.split('from_')[1] 
        maybe_a_poi = "{}@{}".format(sender,domain)
        #print "Sender:{}".format(sender)
    elif from_to_email.startswith('to_'):
        receiver = from_to_email.split('to_')[1]
        maybe_a_poi = "{}@{}".format(receiver,domain)
        #print "Receiver:{}".format(receiver)

    deleted_emails = count_deleted_emails(file_path)

    for key, value in data_dict.items():
        if value.get('email_address') is not None:
            if value['email_address'] == maybe_a_poi:
                # print('poi:{}, poi_deleted_emails:{}'.format(maybe_a_poi,deleted_emails))
                data_dict[key]['poi_deleted_emails'] = deleted_emails
            else:
                data_dict[key]['poi_deleted_emails'] = 0                   
    
def count_deleted_emails(filename):
    deleted = 0
    with open(filename, 'r') as f:
        for line in f:
            items = line.split('deleted_items')
            if len(items) > 1:
                poi = items[0].split('/maildir/')[1]
                deleted += 1    
    return deleted
       
def is_poi(possible_poi):
    for poi in POI_EMAILS:
        if possible_poi.lower() == poi.lower():
            return True
    return False
 
process_emails()            

my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Adding Feature Selection with Select KBest
from sklearn.feature_selection import SelectKBest, chi2
sel_kbest = SelectKBest(score_func=chi2,k=5) # selector instance with chi2 metric
sel_kbest.fit(features,labels) # calculating best five features

scores = sel_kbest.scores_
print(scores)
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:5])
print "{0} best features: {1}\n".format(5, k_best_features.keys())
features_list = ['poi'] + k_best_features.keys()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Using GridSearchCV to do tuning Hyperparameter 
from sklearn.model_selection import GridSearchCV
import numpy as np
import time

params = { "penalty": ["l2"],
           "solver": ["newton-cg"],
           "C": np.logspace(-3,3,1),
           "class_weight": ["balanced"],
           "max_iter": np.arange(1,200,1)
          }

grid = GridSearchCV(clf, params, cv=3)
start = time.time()
grid.fit(features_train, labels_train)
# calculate accuracy
acc = grid.score(features_test, labels_test)
print("Grid search accuracy: {:.2f}%".format(acc * 100))
print("Grid search best parameters: {}".format(grid.best_params_))
print("Best Score: {}".format(grid.best_score_))
hyperparameters = grid.best_params_

# modify the classifier with the new params :) 
# Best params 
#  {'penalty': 'l2', 'C': 0.001, 'max_iter': 169, 'solver': 'newton-cg', 'class_weight': 'balanced'}

clf = LogisticRegression(C=0.0001, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=169,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=42,
          solver='newton-cg', tol=0.0001, verbose=0, warm_start=False)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)