from sklearn import linear_model
from get_data import getData

features, labels = getData()

clf = linear_model.Lasso(alpha=0.1)
clf.fit(features)