#!/usr/bin/python

def get_key(x):
    return(x[2],x[0],x[1])

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    res_errors = {}

    ### your code goes here
    for i in range(len(predictions)):
        x = (ages[i][0],net_worths[i][0],net_worths[i][0] - predictions[i][0])
        cleaned_data.append(x)

    sorted_data = sorted(cleaned_data, key=get_key)
    return sorted_data[9:]

