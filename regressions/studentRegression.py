def studentReg(ages_train, net_worths_train):
    ### import the sklearn regression module, create, and train your regression
    ### name your regression reg
    from sklearn import linear_model
    ### your code goes here!
    reg = linear_model.LinearRegression()
    reg.fit(ages_train, net_worths_train)
    
    
    return reg