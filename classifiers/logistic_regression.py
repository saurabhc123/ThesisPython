"""
File: logistic_regression.py
Description: Train logistic regression classifier in transfer learning setting
"""

import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
from sklearn.metrics import f1_score


source_file = 'data/source.txt'
test_file = 'data/source_validation.txt'

source_data = genfromtxt(source_file, delimiter=',')
test_data = genfromtxt(test_file, delimiter=',')

np.random.shuffle(source_data)

 # Define training and test splits
train_source = source_data[:,1:]
train_source_labels = source_data[:,0]

test_source = test_data[:,1:]
test_source_labels = test_data[:,0]

lr = linear_model.LogisticRegression(C=1e5)
lr.fit(train_source, train_source_labels)

predictions = lr.predict(test_source)

print  f1_score(test_source_labels, predictions)
print lr.coef_
