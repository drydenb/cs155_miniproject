import os 
import sys 
import getopt
import random
import numpy as np
from scipy.stats import randint as sp_randint

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation

import xgboost as xgb

################################################################################
# FUNCTION DEFINITIONS
################################################################################

def usage(argv):

    """
    Prints the usage statement for the script. 
    """

    print "usage: ", argv[0], " -t <train_file> -v <test_file>"
    print """
          Options:
          -t [ --train ]    The file containing the training data.
          -v [ --test ]     The file containing the test data. 
          """
    return 

def main(argv):

    """
    The main function of the script. Executed when __name__ == '__main__'. 
    """

    ########################################
    # Parse command line arguments...
    ########################################

    short_args = "ht:v:"
    long_args = ["help", "train=", "test="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_args, long_args)
    except getopt.GetoptError as err:
        print str(err)    # print the err to stdout
        usage(argv)       # print the usage statement
        sys.exit(1)

    train_file = None
    test_file = None
    for o, a in opts:
        if o in ("-h", "--help"):
            usage(argv)
            sys.exit()
        elif o in ("-t", "--train"):
            train_file = a 
        elif o in ("-v", "--test"):
            test_file = a 
        else:
            usage(argv)
            sys.exit(1)

    if train_file is None or test_file is None:
        usage(argv)
        sys.exit(1)

    ########################################
    # Run the model...
    ########################################

    # comment/uncomment to load original training data
    print "Loading training data..."
    train_data = np.loadtxt(train_file, delimiter='|', skiprows=1)
    x_train = train_data[:, :1000];
    y_train = train_data[:, 1000];

    print "Loading test data..."
    test_data = np.loadtxt(test_file, delimiter='|', skiprows=1)

    # initialize the adaboost classifier (defaults to using shallow decision 
    # trees as the weak learner to boost) and optimize with random search
    print "Training adaboost DT..."
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
                             n_estimators=800)
    bdt.fit(x_train, y_train)

    # Set up a bagged decision tree learner
    print "Training bagged DT..."
    bagged = BaggingClassifier(n_estimators=501, max_samples=0.1)
    bagged.fit(x_train, y_train)

    # initialize a random forest classifier 
    print 'Training random forest...'
    rfc = RandomForestClassifier(n_estimators=600,
                                 max_features=31,
                                 min_samples_split=1,
                                 min_samples_leaf=1)
    rfc.fit(x_train, y_train)

    # start xgboost
    print 'Training XGBoost...'
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(test_data)
    # dtest = xgb.DMatrix(x_test, label=y_test)
    param = {'objective':'binary:logistic', 'silent':1 }
    num_round = 50
    bst = xgb.train( param, dtrain, num_round )
    xgbpreds = bst.predict(dtest)
    # labels = dtest.get_label()

    # use an sgd classifier 
    print 'Training SGD Classifier..'
    clf = SGDClassifier(n_iter=100, loss='hinge', alpha=.0003)
    clf.fit(x_train, y_train)

    print 'Combining models and making predictions...'
    predictions1 = bdt.predict(test_data)
    predictions2 = bagged.predict(test_data)
    predictions3 = rfc.predict(test_data)
    predictions4 = [1 if i > 0.5 else 0 for i in xgbpreds]
    predictions5 = clf.predict(test_data)
    predictions = []
    for i in range(1355):
        if (predictions1[i] + 
            predictions2[i] + 
            predictions3[i] + 
            predictions4[i] + 
            predictions5[i]) > 2:
            predictions.append(1)
        else:
            predictions.append(0)

    print "Writing predictions..."
    predictions_dir = '../../predictions/'
    predictions_filename = (str(os.path.splitext(argv[0])[0]) + 
                            '_predictions.csv')
    predictions_filename = os.path.join(predictions_dir, predictions_filename)
    with open(predictions_filename, 'w') as f:
        f.write('Id,Prediction\n')
        num_samples = 1355 
        for i in range(num_samples):
            f.write(str(i+1) + ',' + str(int(predictions[i])) + '\n')

    sys.exit()

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':
    main(sys.argv)



