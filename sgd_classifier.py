import numpy as np
import random
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation

VALIDATION = 1

print "Loading tf-idf weighted data..."
raw_data = np.loadtxt('training_data.txt', delimiter='|', skiprows=1)
x = np.loadtxt('tf_idf.txt', delimiter='|', skiprows=1)
y = raw_data[:, 1000];


print "Loading test data..."
test_data = np.loadtxt('tf_idf_test.txt', delimiter='|', skiprows=1)



# Cross validation:
# partition the data into training and testing sets 
# (this is not done before submission as we train on the whole dataset)

if (VALIDATION):
    test_size_percentage = 0.20            # size of test set as a percentage in [0., 1.]
    seed = random.randint(0, 2**30)        # pseudo-random seed for split 
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
        x, y, test_size=test_size_percentage, random_state=seed)
else:
    x_train = x
    y_train = y

clf = SGDClassifier(n_iter=100, loss='hinge', alpha=.0003)
clf.fit(x_train, y_train)

if (VALIDATION):
    print "Training score..."
    print clf.score(x_train, y_train)
    print "Scoring..."
    score = clf.score(x_test, y_test)
    print score
else:
    print "Writing predictions..."
    predictions = clf.predict(test_data)
    f = open('predictions.csv', 'w')
    f.write('Id,Prediction\n')
    for i in range(1355):
        f.write(str(i+1) + ',' + str(int(predictions[i])) + '\n')
