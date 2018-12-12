#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# how many data points are enron dataset ? 
print "Size of dataset: {}".format(len(enron_data))

# how many features ? 
print "# of Features: {}".format(len(enron_data['METTS MARK']))

# how many person of interest 
poi = set()
for person, values  in enron_data.items():
    if values['poi']:
        poi.add(person)   

print "# of POI: {}".format(len(poi))

# How many poi's exist? 
# $ cat ../final_project/poi_names.txt | wc -l
# 35

# What is the total value of the stock belonging to James Prentice?
name = str.upper('Prentice James')
print "Stock value: {}".format(enron_data[name]['total_stock_value'])

# How many email messages do we have from Wesley Colwell to persons of interest?
name = str.upper('Colwell Wesley')
feature = 'from_this_person_to_poi'
print "Email to POI: {}".format(enron_data[name][feature])

'''
Of these three individuals (Lay, Skilling and Fastow), who took home the most money (largest value of "total_payments" feature)?
How much money did that person get?
'''
feature = 'total_payments'
total_payments = {}

for enron, values in enron_data.items():
   if values[feature] != 'NaN' and enron != 'TOTAL':
      total_payments[enron] = values[feature]

maximum = max(total_payments.values())
for enron, values in enron_data.items():
    if values[feature] == maximum: 
        print "Who:{} how much:{}".format(enron, maximum)
        break

# How many folks in this dataset have a quantified salary? What about a known email address?
feature_1 = 'salary'
feature_2 = 'email_address'
qty_salary = 0
qty_email = 0
for enron, values in enron_data.items():
	if values[feature_1] != 'NaN':
		qty_salary += 1
	if values[feature_2] != 'NaN':
		qty_email += 1

print "# salary: {} email: {}".format(qty_salary,qty_email)
