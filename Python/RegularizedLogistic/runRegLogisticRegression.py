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

    mci_df = pd.read_csv('../../Classification_Table_New.csv', delimiter=',')
    mci_df = mci_df.drop('ID', axis=1)
    Y = mci_df.Conversion.values
    mci_df = mci_df.drop('Conversion', axis=1)

    csf_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                'PTAU_AB142_Ratio', 'Total_TAU', 'TTAU_AB142_Ratio', 'PTAU_TTAU_Ratio']


    X_CSF_ONLY = mci_df[csf_cols].as_matrix()

    classArray = numpy.zeros(len(Y))

    kf = cross_validation.StratifiedKFold(Y, n_folds=10, shuffle=False)
    lg = LogisticRegression(penalty='l1', class_weight='auto', solver='liblinear')
    for train_index, test_index in kf:
        X_train, X_test = X_CSF_ONLY[train_index], X_CSF_ONLY[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        lg.fit(X_train, Y_train)
        predictedClass = lg.predict(X_test)
        classArray[test_index] = predictedClass
        print(confusion_matrix(Y_test, predictedClass))
    print("Final : ")
    print(confusion_matrix(Y, classArray))

if __name__ == '__main__':
    main()
