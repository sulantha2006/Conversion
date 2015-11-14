__author__ = 'Sulantha'
import numpy
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix

class RegularizedLogisticLearner:
    def __init__(self):
        pass

    def trainLogisticRegreion(self):
        pass

def main():

    mci_df = pd.read_csv('DataFiles/CSF_ONLY/SET_1.csv', delimiter=',')
    #mci_df = mci_df.drop('ID', axis=1)
    mci_df_train = mci_df.loc[mci_df['SAMPLE'] == 1]
    mci_df_test = mci_df.loc[mci_df['SAMPLE'] == 2]
    FY_train = mci_df_train.CONV.values
    FY_test = mci_df_test.CONV.values


    DATA_cols = ['AGE_D1', 'GENDER_CODE', 'APOE4_BIN', 'ABETA', 'PTAU', 'TAU', 'PTAU_TAU', 'PTAU_ABETA'
                 , 'TAU_ABETA']


    X_CSF_ONLY_train = mci_df_train[DATA_cols].as_matrix()
    X_CSF_ONLY_test = mci_df_test[DATA_cols].as_matrix()

    classArray_train = numpy.zeros(len(FY_train))
    classArray_test = numpy.zeros((len(FY_test), 10))

    kf = cross_validation.StratifiedKFold(FY_train, n_folds=10, shuffle=False)
    lg = LogisticRegression(penalty='l1', class_weight='auto', solver='liblinear')
    fold = 0
    for train_index, test_index in kf:
        X_train, X_test = X_CSF_ONLY_train[train_index], X_CSF_ONLY_train[test_index]
        Y_train, Y_test = FY_train[train_index], FY_train[test_index]
        lg.fit(X_train, Y_train)
        predictedClass = lg.predict(X_test)
        classArray_train[test_index] = predictedClass
        classArray_test[:,fold] = lg.predict(X_CSF_ONLY_test)
        print(confusion_matrix(Y_test, predictedClass))
        fold+=1
    print("Validation Final : ")
    print(confusion_matrix(FY_train, classArray_train))

    classArray_test_final = numpy.mean(classArray_test, axis=1) > 0.5
    print("Test Final : ")
    print(confusion_matrix(FY_test, classArray_test_final))






if __name__ == '__main__':
    main()
