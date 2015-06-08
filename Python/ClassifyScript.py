__author__ = 'sulantha'
import numpy
from sklearn.ensemble import RandomForestClassifier

def main():
    dataset = numpy.genfromtxt('/home/sulantha/PycharmProjects/Conversion/Classification_Table.csv', delimiter=',', usecols=range(1, 20), dtype='f8')[1:]
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]

    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(train, target)


if __name__=="__main__":
    main()