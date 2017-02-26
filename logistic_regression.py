"""
File: logistic_regression.py
Description: Train logistic regression classifier in transfer learning setting
"""

import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
from sklearn.metrics import f1_score


source_file = '/Users/sneha/Documents/dev/ThesisPython/source.txt'

source_data = genfromtxt(source_file, delimiter=',')
np.random.shuffle(source_data)

 # Define training and test splits
train_source = source_data[:150,1:]
train_source_labels = source_data[:150,0]

test_source = source_data[151:,1:]
test_source_labels = source_data[151:,0]

lr = linear_model.LogisticRegression(C=1e5)
lr.fit(train_source, train_source_labels)

print lr.score(test_source, test_source_labels)
print lr.coef_
