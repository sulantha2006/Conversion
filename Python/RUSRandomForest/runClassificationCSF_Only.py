__author__ = 'sulantha'
import pandas as pd
import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from Python.RUSRandomForest import RUSRandomForestClassifier
from Python.RUSRandomForest import Config
from multiprocessing import Pool


def writeSensAndSpec(fpr, tpr, thresh, filename):
    specificity = 1 - fpr
    a = numpy.vstack([specificity, tpr, thresh])
    b = numpy.transpose(a)
    numpy.savetxt(filename, b, fmt='%.5f', delimiter=',')


def doRUSRFC(analysisDict):
    print('{0} Started'.format(analysisDict['analysisName']))
    RUSRFC = RUSRandomForestClassifier.RUSRandomForestClassifier(n_Forests=200, n_TreesInForest=300)
    predClasses, classProb, featureImp, featureImpSD = RUSRFC.CVJungle(analysisDict['X_train'], analysisDict['Y_train'],
                                                                       shuffle=True, print_v=True, k=10)
    cm = confusion_matrix(analysisDict['Y_train'], predClasses)
    print('Analysis - {0}  - Validation - \n{1}'.format(analysisDict['analysisName'], cm))

    #### For test set
    predClasses_test = RUSRFC.predict(analysisDict['X_test'])
    cm_test = confusion_matrix(analysisDict['Y_test'], predClasses_test)
    print('Test set results - \n{0}'.format(cm_test))
    ####
    #### For test set
    predClassesProb_test = RUSRFC.predict_prob(analysisDict['X_test'])
    false_positive_rate_test, true_positive_rate_test, thresholds_test = roc_curve(analysisDict['Y_test'], predClassesProb_test[:, 1])
    writeSensAndSpec(false_positive_rate_test, true_positive_rate_test, thresholds_test,
                     Config.figOutputPath + analysisDict['specificitySensitivityFile_test'])
    roc_auc_test = auc(false_positive_rate_test, true_positive_rate_test)
    ####


    featureImpScale = [featureImp[analysisDict['analysis_cols'].index(i)] if i in analysisDict['analysis_cols'] else 0
                       for i in analysisDict['all_list']]
    featureImpScaleSD = [
        featureImpSD[analysisDict['analysis_cols'].index(i)] if i in analysisDict['analysis_cols'] else 0 for i in
        analysisDict['all_list']]

    plt.figure()
    plt.title('Feature Importance')
    plt.bar(range(len(analysisDict['all_list'])), featureImpScale, alpha=0.4, edgecolor="none", linewidth=0.0, color='g',
            align='center', orientation='vertical', width=0.7)
    plt.xticks(range(len(analysisDict['all_list'])), [Config.xticks_dict[tick] for tick in analysisDict['all_list']])
    plt.xticks(rotation=55, ha='right')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(Config.figOutputPath + analysisDict['featureImpFileName'])

    false_positive_rate, true_positive_rate, thresholds = roc_curve(analysisDict['Y_train'], classProb[:, 1])
    writeSensAndSpec(false_positive_rate, true_positive_rate, thresholds,
                     Config.figOutputPath + analysisDict['specificitySensitivityFile'])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    r_dict = dict(analysisName=analysisDict['analysisName'], fpr=false_positive_rate, tpr=true_positive_rate, th=thresholds, auc=roc_auc,
                  fpr_test=false_positive_rate_test, tpr_test=true_positive_rate_test, th_test=thresholds_test, auc_test=roc_auc_test)
    return r_dict


def main():
    mci_df = pd.read_csv('DataFiles/CSF_ONLY/SET_1.csv', delimiter=',')
    mci_df_train = mci_df.loc[mci_df['SAMPLE'] == 1]
    mci_df_test = mci_df.loc[mci_df['SAMPLE'] == 2]
    #mci_df = mci_df.drop('ID', axis=1)
    Y_train = mci_df_train.CONV.values
    Y_test = mci_df_test.CONV.values
    mci_df = mci_df.drop('CONV', axis=1)

    DATA_cols = ['AGE_D1', 'GENDER_CODE', 'APOE4_BIN', 'ABETA', 'PTAU', 'TAU', 'PTAU_TAU', 'PTAU_ABETA'
                 , 'TAU_ABETA']

    X_DATA_ONLY_train = mci_df_train[DATA_cols].as_matrix()
    X_DATA_ONLY_test = mci_df_test[DATA_cols].as_matrix()

    FPRDict = {}
    TPRDict = {}
    ThreshDict = {}
    AUCDict = {}
    FPRDict_Test = {}
    TPRDict_Test = {}
    ThreshDict_Test = {}
    AUCDict_Test = {}

    itemList = [dict(X_train=X_DATA_ONLY_train, Y_train=Y_train, X_test=X_DATA_ONLY_test, Y_test=Y_test, analysisName='DATA_ANALYSIS', analysis_cols=DATA_cols, all_list=DATA_cols,
                     featureImpFileName='CSF_ONLY_FEATURE_IMP.png',
                     specificitySensitivityFile='CSF_ONLY_SensSpec.csv', specificitySensitivityFile_test='CSF_ONLY_SensSpec_testSet.csv'),
                ]

    pool = Pool(processes=6)
    results = pool.map(doRUSRFC, itemList)
    for result in results:
        FPRDict[result['analysisName']] = result['fpr']
        TPRDict[result['analysisName']] = result['tpr']
        ThreshDict[result['analysisName']] = result['th']
        AUCDict[result['analysisName']] = result['auc']
        FPRDict_Test[result['analysisName']] = result['fpr_test']
        TPRDict_Test[result['analysisName']] = result['tpr_test']
        ThreshDict_Test[result['analysisName']] = result['th_test']
        AUCDict_Test[result['analysisName']] = result['auc_test']

    pool.close()
    pool.join()

    plt.figure()

    plt.plot(FPRDict['DATA_ANALYSIS'], TPRDict['DATA_ANALYSIS'], 'b', alpha=0.5, lw = 2,
             label='ROC curve  - Validation {0} (area = {1:0.2f})'.format('CSF', AUCDict['DATA_ANALYSIS']))
    plt.plot(FPRDict_Test['DATA_ANALYSIS'], TPRDict_Test['DATA_ANALYSIS'], 'r', alpha=0.5, lw = 2,
             label='ROC curve - Test {0} (area = {1:0.2f})'.format('CSF', AUCDict_Test['DATA_ANALYSIS']))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for MCI Converters')
    plt.legend(loc="lower right", prop={'size':'small', 'family':'sans-serif'})
    plt.tight_layout()
    plt.savefig(Config.figOutputPath + 'full.png')


if __name__ == '__main__':
    main()
