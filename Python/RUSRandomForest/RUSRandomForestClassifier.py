__author__ = 'Sulantha'
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

class RUSRandomForestClassifier:
    __n_Forests = None
    __n_TreesInForest = None
    __jungle = []

    def __init__(self, n_Forests = 100, n_TreesInForest = 200):
        self.__n_Forests = n_Forests
        self.__n_TreesInForest = n_TreesInForest

    def __rusData(self, X, Y):
        classes, classIndexes, classCounts = numpy.unique(Y, return_inverse = True, return_counts = True)
        minCount = numpy.min(classCounts)
        rusX = numpy.array([])
        rusY = numpy.array([])
        for classIdx in range(len(classes)):
            classSampleIdx = numpy.random.choice(classCounts[classIdx], minCount, replace=False)
            classSampleX = X[classIndexes==classIdx][classSampleIdx]
            classSampleY = Y[classIndexes==classIdx][classSampleIdx]
            rusX = numpy.vstack([rusX, classSampleX]) if rusX.size else classSampleX
            rusY = numpy.append([rusY], [classSampleY]) if rusY.size else classSampleY
        finalMixIdx = numpy.random.choice(len(rusY), len(rusY), replace=False)
        finalRUSX = rusX[finalMixIdx]
        finalRUSY = rusY[finalMixIdx]
        return finalRUSX, finalRUSY

    def __trainForest(self, X, Y):
        rf = RandomForestClassifier(n_estimators=self.__n_TreesInForest, n_jobs = -1, verbose = 0, class_weight = 'auto')
        rfc = rf.fit(X, Y)
        return rfc

    def trainJungle(self, X, Y):
        for i in range(self.__n_Forests):
            X_t, Y_t = self.__rusData(X, Y)
            rf = self.__trainForest(X_t, Y_t)
            self.__jungle.insert(i, rf)

    def predict(self, X):
        forestClass = numpy.array([])
        for forest in self.__jungle:
            forestClass = numpy.vstack([forestClass, forest.predict(X)]) if forestClass.size else forest.predict(X)
        return 1*(numpy.mean(forestClass, axis=0)>0.5)

    def predict_prob(self, X):
        forestClassProb = numpy.array([])
        for forest in self.__jungle:
            forestClassProb = numpy.dstack([forestClassProb, forest.predict_proba(X)]) if forestClassProb.size else forest.predict_proba(X)
        return numpy.mean(forestClassProb, axis=2)

    def CVJungle(self, X, Y, method='kFold', k = 10):
        kf = cross_validation.KFold(len(Y), n_folds=k)
        n_samples = len(Y)
        n_classes = len(numpy.unique(Y))
        classArray = numpy.zeros(n_samples)
        probArray = numpy.zeros((n_samples, n_classes))

        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            self.trainJungle(X_train, Y_train)

            classArray[test_index] = self.predict(X_test)
            probArray[test_index,:] = self.predict_prob(X_test)

        return classArray, probArray


