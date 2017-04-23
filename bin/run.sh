#!/bin/bash

# This script contains a sample run that uses each model to 
# generate predictions. Change parameters as appropriate

# get the directory of this script
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# run each of the models in turn:
echo "Now running all models..."
python $PROJECT_DIR/models/dtree/dtree_adaboost.py \
       -t $PROJECT_DIR/data/raw/training_data.txt \
       -v $PROJECT_DIR/data/raw/testing_data.txt

python $PROJECT_DIR/models/dtree/dtree_bag.py \
       -t $PROJECT_DIR/data/raw/training_data.txt \
       -v $PROJECT_DIR/data/raw/testing_data.txt

python $PROJECT_DIR/models/dtree/dtree_ensemble.py \
       -t $PROJECT_DIR/data/raw/training_data.txt \
       -v $PROJECT_DIR/data/raw/testing_data.txt

python $PROJECT_DIR/models/dtree/dtree_rand_forest.py \
       -t $PROJECT_DIR/data/raw/training_data.txt \
       -v $PROJECT_DIR/data/raw/testing_data.txt

python $PROJECT_DIR/models/sgd/sgd_classifier.py \
       -t $PROJECT_DIR/data/raw/training_data.txt \
       -v $PROJECT_DIR/data/raw/testing_data.txt

# move predictions to the predictions folder
mv $PROJECT_DIR/models/dtree/*.csv $PROJECT_DIR/predictions
mv $PROJECT_DIR/models/sgd/*.csv $PROJECT_DIR/predictions

echo "Finished."