import os
import sys
import getopt
import random
import numpy as np

from sklearn.linear_model import SGDClassifier

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
    y_train = train_data[:, 1000];

    print "Loading test data..."
    test_data = np.loadtxt(test_file, delimiter='|', skiprows=1)

    print "Training SGD classifier..."
    clf = SGDClassifier(n_iter=100, loss='hinge', alpha=.0003)
    clf.fit(x_train, y_train)

    print "Writing predictions..."
    predictions = clf.predict(test_data)
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

