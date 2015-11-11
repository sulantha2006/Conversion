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
    RUSRFC = RUSRandomForestClassifier.RUSRandomForestClassifier(n_Forests=200, n_TreesInForest=500)
    predClasses, classProb, featureImp, featureImpSD = RUSRFC.CVJungle(analysisDict['X'], analysisDict['Y'],
                                                                       shuffle=True, print_v=True)
    cm = confusion_matrix(analysisDict['Y'], predClasses)
    print('Analysis - {0} - {1}'.format(analysisDict['analysisName'], cm))
    featureImpScale = [featureImp[analysisDict['analysis_cols'].index(i)] if i in analysisDict['analysis_cols'] else 0
                       for i in analysisDict['all_list']]
    featureImpScaleSD = [
        featureImpSD[analysisDict['analysis_cols'].index(i)] if i in analysisDict['analysis_cols'] else 0 for i in
        analysisDict['all_list']]

    plt.figure()
    plt.title('Feature Importance {0}'.format(analysisDict['analysisName']))
    plt.bar(range(len(analysisDict['all_list'])), featureImpScale, color='r', align='center', orientation='vertical')
    plt.xticks(range(len(analysisDict['all_list'])), [Config.xticks_dict[tick] for tick in analysisDict['all_list']])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(Config.figOutputPath + analysisDict['featureImpFileName'])

    false_positive_rate, true_positive_rate, thresholds = roc_curve(analysisDict['Y'], classProb[:, 1])
    writeSensAndSpec(false_positive_rate, true_positive_rate, thresholds,
                     Config.figOutputPath + analysisDict['specificitySensitivityFile'])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    r_dict = dict(analysisName=analysisDict['analysisName'], fpr=false_positive_rate, tpr=true_positive_rate, th=thresholds, auc=roc_auc)
    return r_dict


def main():
    mci_df = pd.read_csv('/data/data03/sulantha/MCI_Classification_HAI2016/TESTSETS/SET_1.csv', delimiter=',')
    #mci_df = mci_df.drop('ID', axis=1)
    Y = mci_df.CONV.values
    mci_df = mci_df.drop('CONV', axis=1)

    av45_cols = ['AGE_AV45_D1', 'GENDER_Code', 'APOE_BIN', 'AV45_SUVR_R1', 'AV45_SUVR_R2', 'AV45_SUVR_R3', 'AV45_SUVR_R4', 'AV45_SUVR_R5'
                 , 'AV45_SUVR_R6', 'AV45_SUVR_R7', 'AV45_SUVR_R8', 'AV45_SUVR_R9', 'AV45_GLOBAL_SUVR']

    X_AV45_ONLY = mci_df[av45_cols].as_matrix()

    FPRDict = {}
    TPRDict = {}
    ThreshDict = {}
    AUCDict = {}

    itemList = [dict(X=X_AV45_ONLY, Y=Y, analysisName='AV45_ONLY', analysis_cols=av45_cols, all_list=av45_cols,
                     featureImpFileName='AV45_ONLY_FEATURE_IMP.png',
                     specificitySensitivityFile='AV45_ONLY_SensSpec.csv'),
                ]

    pool = Pool(processes=6)
    results = pool.map(doRUSRFC, itemList)
    for result in results:
        FPRDict[result['analysisName']] = result['fpr']
        TPRDict[result['analysisName']] = result['tpr']
        ThreshDict[result['analysisName']] = result['th']
        AUCDict[result['analysisName']] = result['auc']

    pool.close()
    pool.join()

    plt.figure()

    plt.plot(FPRDict['AV45_ONLY'], TPRDict['AV45_ONLY'], 'b',
             label='ROC curve {0} (area = {1:0.2f})'.format('AV45', AUCDict['AV45_ONLY']))
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
