__author__ = 'sulantha'
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from Python.RUSRandomForest import RUSRandomForestClassifier
from Python.RUSRandomForest import Config

mci_df = pd.read_csv('../../Classification_Table.csv', delimiter=',')
mci_df = mci_df.drop('ID', axis=1)
Y = mci_df.Conversion.values
mci_df = mci_df.drop('Conversion', axis=1)

csf_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos', 'PTAU_AB142_Ratio']
av45_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'AV45_bl_Global_SUVR_NEW', 'AV45_region1', 'AV45_region2',
             'AV45_region3', 'AV45_region4']
fdg_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'FDG_bl_Global_SUVR_NEW', 'FDG_region1', 'FDG_region2', 'FDG_region3',
            'FDG_region4', 'FDG_region5']
csf_av45_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                 'PTAU_AB142_Ratio', 'AV45_bl_Global_SUVR_NEW', 'AV45_region1', 'AV45_region2',
                 'AV45_region3', 'AV45_region4']
csf_fdg_cols = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                'PTAU_AB142_Ratio', 'FDG_bl_Global_SUVR_NEW', 'FDG_region1', 'FDG_region2', 'FDG_region3',
                'FDG_region4', 'FDG_region5']
all_list = ['Age_bl', 'PTGENDER', 'APOE_bin', 'PTAU181P_bl', 'PTAU_Pos', 'ABETA142', 'ABETA142_Pos',
                 'PTAU_AB142_Ratio', 'AV45_bl_Global_SUVR_NEW', 'FDG_bl_Global_SUVR_NEW', 'AV45_region1', 'AV45_region2',
                 'AV45_region3', 'AV45_region4', 'FDG_region1', 'FDG_region2', 'FDG_region3',
                'FDG_region4', 'FDG_region5']

X_CSF_ONLY = mci_df[csf_cols].as_matrix()
X_AV45_ONLY = mci_df[av45_cols].as_matrix()
X_FDG_ONLY = mci_df[fdg_cols].as_matrix()
X_CSF_AV45 = mci_df[csf_av45_cols].as_matrix()
X_CSF_FDG = mci_df[csf_fdg_cols].as_matrix()
X_ALL = mci_df.as_matrix()

print('CSF_ONLY')
RUSRFC_CSF_ONLY = RUSRandomForestClassifier.RUSRandomForestClassifier(n_Forests=200, n_TreesInForest=200)
predClasses_CSF_ONLY, classProb_CSF_ONLY, featureImp_CSF_ONLY, featureImpSD_CSF_ONLY = RUSRFC_CSF_ONLY.CVJungle(X_CSF_ONLY, Y, shuffle=True, print_v=True)
cm_CSF_ONLY = confusion_matrix(Y, predClasses_CSF_ONLY)
print('Final Accuracy')
print(cm_CSF_ONLY)
print(featureImp_CSF_ONLY)
featureImpScale_CSF_ONLY = [featureImp_CSF_ONLY[csf_cols.index(i)] if i in csf_cols else 0 for i in all_list]
plt.figure()
plt.title('Feature Importance CSF ONLY')
plt.bar(range(len(all_list)), featureImpScale_CSF_ONLY, color='r', align='center', orientation='vertical')
plt.xticks(range(len(all_list)), all_list)
plt.xticks(rotation=90)
plt.savefig(Config.figOutputPath+'CSF_ONLY_FEATURE_IMP.png')
false_positive_rate_CSF_ONLY, true_positive_rate_CSF_ONLY, thresholds_CSF_ONLY = roc_curve(Y, classProb_CSF_ONLY[:, 1])
roc_auc_CSF_ONLY = auc(false_positive_rate_CSF_ONLY, true_positive_rate_CSF_ONLY)

print('AV45_ONLY')
RUSRFC_AV45_ONLY = RUSRandomForestClassifier.RUSRandomForestClassifier(n_Forests=100, n_TreesInForest=500)
predClasses_AV45_ONLY, classProb_AV45_ONLY, featureImp_AV45_ONLY, featureImpSD_AV45_ONLY = RUSRFC_AV45_ONLY.CVJungle(X_AV45_ONLY, Y, shuffle=True, print_v=True)
cm_AV45_ONLY = confusion_matrix(Y, predClasses_AV45_ONLY)
print('Final Accuracy')
print(cm_AV45_ONLY)
print(featureImp_AV45_ONLY)
featureImpScale_AV45_ONLY = [featureImp_AV45_ONLY[av45_cols.index(i)] if i in av45_cols else 0 for i in all_list]
plt.figure()
plt.title('Feature Importance AV45 ONLY')
plt.bar(range(len(all_list)), featureImpScale_AV45_ONLY, color='r', align='center', orientation='vertical')
plt.xticks(range(len(all_list)), all_list)
plt.xticks(rotation=90)
plt.savefig(Config.figOutputPath+'AV45_ONLY_FEATURE_IMP.png')
false_positive_rate_AV45_ONLY, true_positive_rate_AV45_ONLY, thresholds_AV45_ONLY = roc_curve(Y,classProb_AV45_ONLY[:, 1])
roc_auc_AV45_ONLY = auc(false_positive_rate_AV45_ONLY, true_positive_rate_AV45_ONLY)

print('FDG_ONLY')
RUSRFC_FDG_ONLY = RUSRandomForestClassifier.RUSRandomForestClassifier(n_Forests=100, n_TreesInForest=500)
predClasses_FDG_ONLY, classProb_FDG_ONLY, featureImp_FDG_ONLY, featureImpSD_FDG_ONLY = RUSRFC_FDG_ONLY.CVJungle(X_FDG_ONLY, Y, shuffle=True, print_v=True)
cm_FDG_ONLY = confusion_matrix(Y, predClasses_FDG_ONLY)
print('Final Accuracy')
print(cm_FDG_ONLY)
print(featureImp_FDG_ONLY)
featureImpScale_FDG_ONLY = [featureImp_FDG_ONLY[fdg_cols.index(i)] if i in fdg_cols else 0 for i in all_list]
plt.figure()
plt.title('Feature Importance FDG ONLY')
plt.bar(range(len(all_list)), featureImpScale_FDG_ONLY, color='r', align='center', orientation='vertical')
plt.xticks(range(len(all_list)), all_list)
plt.xticks(rotation=90)
plt.savefig(Config.figOutputPath+'FDG_ONLY_FEATURE_IMP.png')
false_positive_rate_FDG_ONLY, true_positive_rate_FDG_ONLY, thresholds_FDG_ONLY = roc_curve(Y, classProb_FDG_ONLY[:, 1])
roc_auc_FDG_ONLY = auc(false_positive_rate_FDG_ONLY, true_positive_rate_FDG_ONLY)

print('CSF_AV45')
RUSRFC_CSF_AV45 = RUSRandomForestClassifier.RUSRandomForestClassifier(n_Forests=100, n_TreesInForest=500)
predClasses_CSF_AV45, classProb_CSF_AV45, featureImp_CSF_AV45, featureImpSD_CSF_AV45 = RUSRFC_CSF_AV45.CVJungle(X_CSF_AV45, Y, shuffle=True, print_v=True)
cm_CSF_AV45 = confusion_matrix(Y, predClasses_CSF_AV45)
print('Final Accuracy')
print(cm_CSF_AV45)
print(featureImp_CSF_AV45)
featureImpScale_CSF_AV45 = [featureImp_CSF_AV45[csf_av45_cols.index(i)] if i in csf_av45_cols else 0 for i in all_list]
plt.figure()
plt.title('Feature Importance CSF & AV45')
plt.bar(range(len(all_list)), featureImpScale_CSF_AV45, color='r', align='center', orientation='vertical')
plt.xticks(range(len(all_list)), all_list)
plt.xticks(rotation=90)
plt.savefig(Config.figOutputPath+'CSF_AV45_FEATURE_IMP.png')
false_positive_rate_CSF_AV45, true_positive_rate_CSF_AV45, thresholds_CSF_AV45 = roc_curve(Y, classProb_CSF_AV45[:, 1])
roc_auc_CSF_AV45 = auc(false_positive_rate_CSF_AV45, true_positive_rate_CSF_AV45)

print('CSF_FDG')
RUSRFC_CSF_FDG = RUSRandomForestClassifier.RUSRandomForestClassifier(n_Forests=100, n_TreesInForest=500)
predClasses_CSF_FDG, classProb_CSF_FDG, featureImp_CSF_FDG, featureImpSD_CSF_FDG = RUSRFC_CSF_FDG.CVJungle(X_CSF_FDG, Y, shuffle=True, print_v=True)
cm_CSF_FDG = confusion_matrix(Y, predClasses_CSF_FDG)
print('Final Accuracy')
print(cm_CSF_FDG)
print(featureImp_CSF_FDG)
featureImpScale_CSF_FDG = [featureImp_CSF_FDG[csf_fdg_cols.index(i)] if i in csf_fdg_cols else 0 for i in all_list]
plt.figure()
plt.title('Feature Importance CSF & FDG')
plt.bar(range(len(all_list)), featureImpScale_CSF_FDG, color='r', align='center', orientation='vertical')
plt.xticks(range(len(all_list)), all_list)
plt.xticks(rotation=90)
plt.savefig(Config.figOutputPath+'CSF_FDG_FEATURE_IMP.png')
false_positive_rate_CSF_FDG, true_positive_rate_CSF_FDG, thresholds_CSF_FDG = roc_curve(Y, classProb_CSF_FDG[:, 1])
roc_auc_CSF_FDG = auc(false_positive_rate_CSF_FDG, true_positive_rate_CSF_FDG)

print('ALL')
RUSRFC_ALL = RUSRandomForestClassifier.RUSRandomForestClassifier(n_Forests=100, n_TreesInForest=500)
predClasses_ALL, classProb_ALL, featureImp_ALL, featureImpSD_ALL = RUSRFC_ALL.CVJungle(X_ALL, Y, shuffle=True, print_v=True)
cm_ALL = confusion_matrix(Y, predClasses_ALL)
print('Final Accuracy')
print(cm_ALL)
print(featureImp_ALL)
featureImpScale_ALL = [featureImp_ALL[all_list.index(i)] if i in all_list else 0 for i in all_list]
plt.figure()
plt.title('Feature Importance ALL VARS')
plt.bar(range(len(all_list)), featureImpScale_ALL, color='r', align='center', orientation='vertical')
plt.xticks(range(len(all_list)), all_list)
plt.xticks(rotation=90)
plt.savefig(Config.figOutputPath+'ALL_FEATURE_IMP.png')
false_positive_rate_ALL, true_positive_rate_ALL, thresholds_ALL = roc_curve(Y, classProb_ALL[:, 1])
roc_auc_ALL = auc(false_positive_rate_ALL, true_positive_rate_ALL)

plt.figure()
plt.plot(false_positive_rate_CSF_ONLY, true_positive_rate_CSF_ONLY, 'g',
         label='ROC curve {0} (area = {1:0.2f})'.format('CSF', roc_auc_CSF_ONLY))
plt.plot(false_positive_rate_AV45_ONLY, true_positive_rate_AV45_ONLY, 'b',
         label='ROC curve {0} (area = {1:0.2f})'.format('AV45', roc_auc_AV45_ONLY))
plt.plot(false_positive_rate_AV45_ONLY, true_positive_rate_FDG_ONLY, 'c',
         label='ROC curve {0} (area = {1:0.2f})'.format('FDG', roc_auc_FDG_ONLY))
plt.plot(false_positive_rate_CSF_AV45, true_positive_rate_CSF_AV45, 'm',
         label='ROC curve {0} (area = {1:0.2f})'.format('CSF & AV45', roc_auc_CSF_AV45))
plt.plot(false_positive_rate_CSF_FDG, true_positive_rate_CSF_FDG, 'y',
         label='ROC curve {0} (area = {1:0.2f})'.format('CSF & FDG', roc_auc_CSF_FDG))
plt.plot(false_positive_rate_ALL, true_positive_rate_ALL, 'r',
         label='ROC curve {0} (area = {1:0.2f})'.format('ALL', roc_auc_ALL))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for MCI Converters')
plt.legend(loc="lower right")
plt.savefig(Config.figOutputPath+'full.png')
