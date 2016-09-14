import os
import sys
import getopt
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

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
    # Perform the conversion...
    ########################################

    # load the data file 
    print "Loading training data..."
    data = np.loadtxt(train_file, delimiter='|', skiprows=1)
    x = data[:, :1000]
    y = data[:, 1000]

    # using the transformer, normalize x using tf-idf weighting 
    print "Normalizing training data with tf-idf weighting..."
    transformer = TfidfTransformer()
    tfidf_train = transformer.fit_transform(x)

    # print out the normalized data to file 
    print "Writing converted training data to file..."
    data_dir = '../tf_idf/'
    train_file_base = os.path.split(train_file)[1]
    train_file_prefix = os.path.splitext(train_file_base)[0]
    tfidf_train_filename = os.path.join(data_dir, train_file_prefix + '_tf_idf.txt')
    np.savetxt(tfidf_train_filename, 
               tfidf_train.toarray(), 
               delimiter='|', 
               header=("tf-idf weighting of " + train_file_base))

    # perform the same procedure with the testing data

    print "Loading test data..."
    test_data = np.loadtxt(test_file, delimiter='|', skiprows=1)

    print "Normalizing test data with tf-idf weighting..."
    transformer2 = TfidfTransformer()
    tfidf_test = transformer2.fit_transform(test_data)

    print "Writing converted test data to file..."
    data_dir = '../tf_idf/'
    test_file_base = os.path.split(test_file)[1]
    test_file_prefix = os.path.splitext(test_file_base)[0]
    tfidf_test_filename = os.path.join(data_dir, test_file_prefix + '_tf_idf.txt')
    np.savetxt(tfidf_test_filename, 
               tfidf_test.toarray(), 
               delimiter='|', 
               header=("tf-idf weighting of " + test_file_base))

################################################################################
# MAIN
################################################################################

if __name__ == '__main__':
    main(sys.argv)



