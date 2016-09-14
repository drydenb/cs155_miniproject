import os
import sys
import getopt 
import numpy as np

from sklearn.grid_search import RandomizedSearchCV   
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from scipy.stats import randint as sp_randint

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

    print "Loading training data..."
    train_data = np.loadtxt(train_file, delimiter='|', skiprows=1)
    x_train = train_data[:, :1000]
    y_train = train_data[:, 1000]

    print "Loading test data..."
    test_data = np.loadtxt(test_file, delimiter='|', skiprows=1)

    # initialize the adaboost classifier (defaults to using shallow decision 
    # trees as the weak learner to boost), optimize params w/ random search
    print "Initializing BDT..."
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))

    # note: you cannot select parameters inside the internal estimator, i.e. 
    # max_depth inside the decision tree won't work (it only optimizes the 
    # parameters on the level of random search)
    print "Training BDT and optimizing parameter space with random search..."
    param_dist = {"n_estimators": sp_randint(200,600)}
    n_iter_search = 2    
    random_search = RandomizedSearchCV(bdt, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    random_search.fit(x_train, y_train)

    # record the predictions to a file 
    print "Writing predictions..."
    predictions = random_search.predict(test_data)
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




