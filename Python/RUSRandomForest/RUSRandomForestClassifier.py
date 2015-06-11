__author__ = 'Sulantha'
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix


class RUSRandomForestClassifier:

    def __init__(self, n_Forests=100, n_TreesInForest=200):
        self.__n_Forests = n_Forests
        self.__n_TreesInForest = n_TreesInForest
        self.__jungle = []

    def __rusData(self, X, Y):
        classes, classIndexes, classCounts = numpy.unique(Y, return_inverse=True, return_counts=True)
        minCount = numpy.min(classCounts)
        rusX = numpy.array([])
        rusY = numpy.array([])
        for classIdx in range(len(classes)):
            classSampleIdx = numpy.random.choice(classCounts[classIdx], minCount, replace=False)
            classSampleX = X[classIndexes == classIdx][classSampleIdx]
            classSampleY = Y[classIndexes == classIdx][classSampleIdx]
            rusX = numpy.vstack([rusX, classSampleX]) if rusX.size else classSampleX
            rusY = numpy.append([rusY], [classSampleY]) if rusY.size else classSampleY
        finalMixIdx = numpy.random.choice(len(rusY), len(rusY), replace=False)
        finalRUSX = rusX[finalMixIdx]
        finalRUSY = rusY[finalMixIdx]
        return finalRUSX, finalRUSY

    def __trainForest(self, X, Y):
        rf = RandomForestClassifier(n_estimators=self.__n_TreesInForest, n_jobs=-1, verbose=0, class_weight='auto')
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
        return 1 * (numpy.mean(forestClass, axis=0) > 0.5)

    def predict_prob(self, X):
        forestClassProb = numpy.array([])
        for forest in self.__jungle:
            forestClassProb = numpy.dstack(
                [forestClassProb, forest.predict_proba(X)]) if forestClassProb.size else forest.predict_proba(X)
        return numpy.mean(forestClassProb, axis=2)

    def featureImpotance(self):
        feat_imp_arr = numpy.array([])
        std_arr = numpy.array([])
        for forest in self.__jungle:
            feat_imp_arr = numpy.vstack(
                [feat_imp_arr, forest.feature_importances_]) if feat_imp_arr.size else forest.feature_importances_
            std = numpy.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
            std_arr = numpy.vstack([std_arr, std]) if std_arr.size else std
        return numpy.mean(feat_imp_arr, axis=0), numpy.mean(std_arr, axis=0)

    def CVJungle(self, X, Y, method='stratified', k=10, shuffle=False, print_v=False):
        n_samples = len(Y)
        n_classes = len(numpy.unique(Y))
        classArray = numpy.zeros(n_samples)
        probArray = numpy.zeros((n_samples, n_classes))
        if method == 'kFold':
            kf = cross_validation.KFold(len(Y), n_folds=k, shuffle=shuffle)
        elif method == 'stratified':
            kf = cross_validation.StratifiedKFold(Y, n_folds=k, shuffle=shuffle)
        featureImpArray = numpy.array([])
        featureImpStdArray = numpy.array([])
        for train_index, test_index in kf:
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            self.trainJungle(X_train, Y_train)

            predictedClass = self.predict(X_test)
            classArray[test_index] = predictedClass

            probArray[test_index, :] = self.predict_prob(X_test)
            if print_v: print(confusion_matrix(Y_test, predictedClass))

            featureImpMean, featureImpSd = self.featureImpotance()
            featureImpArray = numpy.vstack(
                [featureImpArray, featureImpMean]) if featureImpArray.size else featureImpMean
            featureImpStdArray = numpy.vstack([featureImpStdArray, featureImpSd]) if featureImpStdArray.size else featureImpSd

        return classArray, probArray, numpy.mean(featureImpArray, axis=0), numpy.mean(featureImpStdArray, axis=0)
