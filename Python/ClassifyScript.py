__author__ = 'sulantha'
import numpy
from sklearn.ensemble import RandomForestClassifier, partial_dependence
from sklearn.cross_validation import *

def main():
    dataset = numpy.genfromtxt('../Classification_Table.csv', delimiter=',', usecols=range(1, 20), dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]

    rf = RandomForestClassifier(n_estimators=500, n_jobs = -1, verbose = 1, class_weight = 'auto')
    rf.fit(train, target)

    cv = StratifiedKFold(target, 10)
    scores = cross_val_score(rf, train, target, cv=cv)
    print("Accuracy: %0.2f (+/- %0.2f)"
      % (scores.mean(), scores.std()*2))
    print(rf.feature_importances_)


if __name__=="__main__":
    main()