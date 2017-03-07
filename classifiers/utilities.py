from sklearn.ensemble import RandomForestClassifier
import numpy as np

def listTopDimensions(data, labels):
    clf = RandomForestClassifier()
    clf.fit(data, labels)

    importances = clf.feature_importances_
    #usefulDimensions = filter(lambda importance: importance > 0.0, importances)
    p = np.argsort(importances)[::-1]
    top_dimensions = []

    for i in xrange(10):
        top_dimensions.append(p[i])
    return top_dimensions