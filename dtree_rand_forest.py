import numpy as np
import matplotlib.pyplot as plt

# for exhaustive parameter search
# from sklearn.grid_search import GridSearchCV  

from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from scipy.stats import randint as sp_randint

# comment/uncomment to load original training data
print "Loading training data..."
raw_data = np.loadtxt('training_data.txt', delimiter='|', skiprows=1)
x = raw_data[:, :1000];
y = raw_data[:, 1000];
# partition the data into training and testing sets 
# (this is not done before submission as we train on the whole dataset)
# test_size_percentage = 0.20    # size of test set as a percentage in [0., 1.]
# seed = 0                       # pseudo-random seed for split 
# x_train, x_test, y_train, y_test = cross_validation.train_test_split(
# 	x, y, test_size=test_size_percentage, random_state=seed)

# for submission purposes only
x_train = x
y_train = y 
print "Loading test data..."
test_data = np.loadtxt('testing_data.txt', delimiter='|', skiprows=1)

# initialize a random forest classifier 
rfc = RandomForestClassifier(n_estimators=20)    # 20 is the default in sklearn example 

# # Utility function to report best scores
# def report(grid_scores, n_top=3):
#     top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
#     for i, score in enumerate(top_scores):
#         print("Model with rank: {0}".format(i + 1))
#         print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#               score.mean_validation_score,
#               np.std(score.cv_validation_scores)))
#         print("Parameters: {0}".format(score.parameters))
#         print("")

# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
print "Training..."
n_iter_search = 20
random_search = RandomizedSearchCV(rfc, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search.fit(x_train, y_train)

print "Writing predictions..."
predictions = random_search.predict(test_data)
f = open('predictions.csv', 'w')
f.write('Id,Prediction\n')
for i in range(1355):
	f.write(str(i+1) + ',' + str(int(predictions[i])) + '\n')
