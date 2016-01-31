import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import random

raw_data = np.loadtxt('training_data.txt', delimiter='|', skiprows=1)
x_train = raw_data[:, :1000];
y_train = raw_data[:, 1000];

'''
x_train = raw_data[:3000, :1000]
y_train = raw_data[:3000, 1000]
x_test = raw_data[3000:, :1000]
y_test = raw_data[3000:, 1000]

training_errors = []
test_errors = []
for i in range(1, 26):
	clf = tree.DecisionTreeClassifier(min_samples_leaf=i)
	clf = clf.fit(x_train, y_train)

	training_errors.append(1 - clf.score(x_train, y_train))
	test_errors.append(1 - clf.score(x_test, y_test))

# Plot the errors
plt.figure(1)
plota1, = plt.plot(range(1, 26), training_errors, label='Training Error')
plota2, = plt.plot(range(1, 26), test_errors, label='Test Error')
plt.title('Error vs. min leaf node size for decision trees')
plt.xlabel('Minimum Leaf Node Size')
plt.ylabel('Error')
plt.legend(handles=[plota1, plota2], loc=1)
plt.show()
'''

# Set up variables
samples = 101
sample_size = 400
clfs = []

# Train some trees
print 'Training trees...'
for i in range(samples):
	x_sub = []
	y_sub = []
	for j in range(sample_size):
		index = random.randint(0, 4189-1)
		x_sub.append(x_train[index])
		y_sub.append(y_train[index])

	# Data to run our test model on
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(x_sub, y_sub)
	clfs.append(clf)

# Create predictions and aggregate
print 'Aggregating predictions...'
test_data = np.loadtxt('testing_data.txt', delimiter='|', skiprows=1)
agg_predictions = clfs[0].predict(test_data)
for i in range(1, samples):
	agg_predictions += clfs[i].predict(test_data)

predictions = []
for i in range(1355):
	if agg_predictions[i] > (samples / 2):
		predictions.append(1)
	else:
		predictions.append(0)

# Write predictions to file
f = open('predictions.csv', 'w')
f.write('Id,Prediction\n')
for i in range(1355):
	f.write(str(i+1) + ',' + str(int(predictions[i])) + '\n')