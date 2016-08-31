import sys 
import getopt 
import numpy as np

# import matplotlib.pyplot as plt 

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
    # Parse command line arguments 
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
    # Load the data
    ########################################

    print "Loading training data..."
    train_data = np.loadtxt(train_file, delimiter='|', skiprows=1)
    x_train = train_data[:, :1000]
    y_train = train_data[:, 1000]

    # partition the data into training and testing sets 
    # (this is not done before submission as we train on the whole dataset)
    # test_size_percentage = 0.20    # size of test set as a percentage in [0., 1.]
    # seed = 0                       # pseudo-random seed for split 
    # x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    #     x, y, test_size=test_size_percentage, random_state=seed)

    # for submission purposes only
    # x_train = x
    # y_train = y 

    print "Loading test data..."
    test_data = np.loadtxt(test_file, delimiter='|', skiprows=1)
    x_test = train_data[:, :1000]
    y_test = train_data[:, 1000]

    # initialize the adaboost classifier (defaults to using shallow decision trees
    # as the weak learner to boost) and optimize parameters using random search
    print "Initializing BDT..."
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))

    print "Training BDT and optimizing parameter space with random search..."
    # it doesn't seem like you can select parameters inside the internal estimator
    # i.e. max_depth inside the decision tree won't work (it only optimizes the parameters 
    # on the level of random search)
    param_dist = {"n_estimators": sp_randint(200,600)}
    n_iter_search = 2    # 20 is the number choosen in the example in sklearn
    random_search = RandomizedSearchCV(bdt, param_distributions=param_dist,
                                       n_iter=n_iter_search)
    random_search.fit(x_train, y_train)

    # sanity checks that the parameters inside random_search were changed 
    # print "Get params:"
    # print random_search.get_params()
    # print "Best estimator:"
    # print random_search.best_estimator_
    # print "Best params:"
    # print random_search.best_params_

    # score the classfier on the test set 
    print "Scoring..."
    print random_search.score(x_test, y_test)

    print "Writing predictions..."
    predictions = random_search.predict(test_data)
    f = open('predictions.csv', 'w')
    f.write('Id,Prediction\n')
    for i in range(1355):
        f.write(str(i+1) + ',' + str(int(predictions[i])) + '\n')

    sys.exit()

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':
    main(sys.argv)

################################################################################
# RESULTS:
################################################################################

# with original training data:
# training with 75%, testing on 25% yields score = 0.662213740458

# with transformed tf-idf data (not sure if this is working yet 
# because this seems signifigantly worse than with the original data):
# training with 75%, testing on 25% yields score = 0.644083969466

# submission using all training data yields about 0.53 on the leaderboard
