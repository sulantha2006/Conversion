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
    mci_df = pd.read_csv('../../Classification_Table_New.csv', delimiter=',')
    mci_df = mci_df.drop('ID', axis=1)
    Y = mci_df.Conversion.values
    mci_df = mci_df.drop('Conversion', axis=1)

    csf_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                'PTAU_AB142_Ratio', 'Total_TAU', 'TTAU_AB142_Ratio', 'PTAU_TTAU_Ratio']
    av45_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'AV45_bl_Global_SUVR_NEW', 'AV45_region1', 'AV45_region2',
                 'AV45_region3', 'AV45_region4']
    fdg_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'FDG_bl_Global_SUVR_NEW', 'FDG_region1', 'FDG_region2', 'FDG_region3',
                'FDG_region4', 'FDG_region5']
    csf_av45_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                     'PTAU_AB142_Ratio', 'Total_TAU', 'TTAU_AB142_Ratio', 'PTAU_TTAU_Ratio', 'AV45_bl_Global_SUVR_NEW', 'AV45_region1', 'AV45_region2',
                     'AV45_region3', 'AV45_region4']
    csf_fdg_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                    'PTAU_AB142_Ratio', 'Total_TAU', 'TTAU_AB142_Ratio', 'PTAU_TTAU_Ratio', 'FDG_bl_Global_SUVR_NEW', 'FDG_region1', 'FDG_region2', 'FDG_region3',
                    'FDG_region4', 'FDG_region5']
    all_list = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                'PTAU_AB142_Ratio', 'Total_TAU', 'TTAU_AB142_Ratio', 'PTAU_TTAU_Ratio', 'AV45_bl_Global_SUVR_NEW', 'FDG_bl_Global_SUVR_NEW', 'AV45_region1', 'AV45_region2',
                'AV45_region3', 'AV45_region4', 'FDG_region1', 'FDG_region2', 'FDG_region3',
                'FDG_region4', 'FDG_region5']

    X_CSF_ONLY = mci_df[csf_cols].as_matrix()
    X_AV45_ONLY = mci_df[av45_cols].as_matrix()
    X_FDG_ONLY = mci_df[fdg_cols].as_matrix()
    X_CSF_AV45 = mci_df[csf_av45_cols].as_matrix()
    X_CSF_FDG = mci_df[csf_fdg_cols].as_matrix()
    X_ALL = mci_df.as_matrix()

    FPRDict = {}
    TPRDict = {}
    ThreshDict = {}
    AUCDict = {}

    itemList = [dict(X=X_CSF_ONLY, Y=Y, analysisName='CSF_ONLY', analysis_cols=csf_cols, all_list=all_list,
                     featureImpFileName='CSF_ONLY_FEATURE_IMP.png', specificitySensitivityFile='CSF_ONLY_SensSpec.csv'),
                dict(X=X_AV45_ONLY, Y=Y, analysisName='AV45_ONLY', analysis_cols=av45_cols, all_list=all_list,
                     featureImpFileName='AV45_ONLY_FEATURE_IMP.png',
                     specificitySensitivityFile='AV45_ONLY_SensSpec.csv'),
                dict(X=X_FDG_ONLY, Y=Y, analysisName='FDG_ONLY', analysis_cols=fdg_cols, all_list=all_list,
                     featureImpFileName='FDG_ONLY_FEATURE_IMP.png', specificitySensitivityFile='FDG_ONLY_SensSpec.csv'),
                dict(X=X_CSF_AV45, Y=Y, analysisName='CSF_AV45', analysis_cols=csf_av45_cols, all_list=all_list,
                     featureImpFileName='CSF_AV45_FEATURE_IMP.png', specificitySensitivityFile='CSF_AV45_SensSpec.csv'),
                dict(X=X_CSF_FDG, Y=Y, analysisName='CSF_FDG', analysis_cols=csf_fdg_cols, all_list=all_list,
                     featureImpFileName='CSF_FDG_FEATURE_IMP.png', specificitySensitivityFile='CSF_FDG_SensSpec.csv'),
                dict(X=X_ALL, Y=Y, analysisName='ALL', analysis_cols=all_list, all_list=all_list,
                     featureImpFileName='ALL_FEATURE_IMP.png', specificitySensitivityFile='ALL_SensSpec.csv')]

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
    plt.plot(FPRDict['CSF_ONLY'], TPRDict['CSF_ONLY'], 'g',
             label='ROC curve {0} (area = {1:0.2f})'.format('CSF', AUCDict['CSF_ONLY']))
    plt.plot(FPRDict['AV45_ONLY'], TPRDict['AV45_ONLY'], 'b',
             label='ROC curve {0} (area = {1:0.2f})'.format('AV45', AUCDict['AV45_ONLY']))
    plt.plot(FPRDict['FDG_ONLY'], TPRDict['FDG_ONLY'], 'c',
             label='ROC curve {0} (area = {1:0.2f})'.format('FDG', AUCDict['FDG_ONLY']))
    plt.plot(FPRDict['CSF_AV45'], TPRDict['CSF_AV45'], 'm',
             label='ROC curve {0} (area = {1:0.2f})'.format('CSF & AV45', AUCDict['CSF_AV45']))
    plt.plot(FPRDict['CSF_FDG'], TPRDict['CSF_FDG'], 'y',
             label='ROC curve {0} (area = {1:0.2f})'.format('CSF & FDG', AUCDict['CSF_FDG']))
    plt.plot(FPRDict['ALL'], TPRDict['ALL'], 'r',
             label='ROC curve {0} (area = {1:0.2f})'.format('ALL', AUCDict['ALL']))
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
