import os 
import sys
import random
import getopt
import numpy as np

from sklearn import tree

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
    test_data = np.loadtxt(test_file, delimiter='|', skiprows=1)
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
    print "Writing predictions..."
    predictions_dir = '../../predictions/'
    predictions_filename = (str(os.path.splitext(argv[0])[0]) + 
                            '_predictions.csv')
    predictions_filename = os.path.join(predictions_dir, predictions_filename)
    with open(predictions_filename, 'w') as f:
        f.write('Id,Prediction\n')
        for i in range(1355):
            f.write(str(i+1) + ',' + str(int(predictions[i])) + '\n')
    
    sys.exit()

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':
    main(sys.argv)



