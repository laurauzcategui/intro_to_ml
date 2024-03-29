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
features_list = ['poi','salary','bonus','long_term_incentive','loan_advances','poi_deleted_emails'] # You will need to use more features

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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)


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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)