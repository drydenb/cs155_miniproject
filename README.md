# Text Classification using scikit-learn

## Introduction

This is a Kaggle project for CS 155 at Caltech on Sentiment Analysis / Text
Classification. 

Each sample in this project corresponds to a speech, pre-processed into a
bag-of-words representation. The goal of the project is to investigate 
various machine learning models to predict opposition (=0) or support (=1)
given some sample. The results of the project as well are summarized in a 
report under `docs/`.

While some of the code has been refactored and the directory tree has 
been restructured by Dryden Bouamalay, this project would not have been
possible without his peers Jeffery An and Rohan Batra.

Project structure: 
* `data/` Contains preprocessing scripts and data files to run the analysis. Under `raw/`, `training_data.txt` contains 4189 speeches used train our models, while `testing_data.txt` contains 1355 test points used for making predictions. 
* `docs/` Contains a PDF summarizing the results of the project.
* `models/` Contains Python scripts which can be run on data files in `data/` to generate .csv files to `predictions/`.
* `predictions/` Contains `.csv` files that comprise model predictions. 

## Run

To test out a model, run Python on the corresponding script in `models/` with
training and test data as arguments, i.e.: 

`python dtree_adaboost.py -t ../../data/raw/training_data.txt -v ../../data/raw/testing_data.txt`

The predictions made by the model will be put in `predictions/`.

