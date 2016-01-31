# implements tf-idf term weighting.

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

# load all the data 
print "Loading training data..."
raw_data = np.loadtxt('training_data.txt', delimiter='|', skiprows=1)
x_train = raw_data[:, :1000];
y_train = raw_data[:, 1000];

# using the transformer, normalize x_train using tf-idf weighting 
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(x_train)

np.savetxt('tf_idf.txt', 
	       tfidf.toarray(), 
	       delimiter='|', 
	       header="tf-idf weighting of 'training_data.txt'")
# TfidfTransformer(norm=...'l2', smooth_idf=True, sublinear_tf=False,
#                  use_idf=True)

