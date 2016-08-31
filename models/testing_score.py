import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation

# load all the data 
print "Loading training data..."
raw_data = np.loadtxt('training_data.txt', delimiter='|', skiprows=1)
x_train = raw_data[:, :1000];
y_train = raw_data[:, 1000];

# load tf_idf weighted data
# print "Loading tf-idf weighted data..."
# raw_data = np.loadtxt('training_data.txt', delimiter='|', skiprows=1)
# x_train = np.loadtxt('tf_idf.txt', delimiter='|', skiprows=1)
# y_train = raw_data[:, 1000];

# print "Loading test data..."
# test_data = np.loadtxt('testing_data.txt', delimiter='|', skiprows=1)
# x_test = test_data[:, :1000];

# estimator_range = [i for i in range(1, 600 + 1)]
# estimators = estimator_range[9::100]
# validation_means = [] 
# initialize the boosted decision tree with the adaboost classifier 
print "Training bdt..."
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
	                     n_estimators=50,
	                     learning_rate=1.)
print "Initialized bdt with n_estimators = " + str(bdt.n_estimators)

bdt.fit(x_train, y_train)
print bdt.get_params()
# print bdt.score(x_train, y_train)

# print "Computing validation mean..."
# scores = cross_validation.cross_val_score(bdt, x_train, y_train, cv=5)

# # fit the boosted decision tree
# # print "Training bdt..."
# # bdt.fit(x_train, y_train)
# # print bdt.predict(x_test)

# print "Generating CV scores..."
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


