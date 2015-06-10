__author__ = 'Sulantha'
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from Python.RUSRandomForest import RUSRandomForestClassifier

mci_df = pd.read_csv('../../Classification_Table.csv', delimiter=',')
mci_df = mci_df.drop('ID', axis=1)
Y = mci_df.Conversion.values
mci_df = mci_df.drop('Conversion', axis=1)
X = mci_df.as_matrix()

RUSRFC = RUSRandomForestClassifier.RUSRandomForestClassifier()
predClasses, classProb = RUSRFC.CVJungle(X, Y, shuffle=True, print_v=True)
cm = confusion_matrix(Y, predClasses)
print('Final Accuracy')
print(cm)

false_positive_rate_1, true_positive_rate_1, thresholds_1 = roc_curve(Y, classProb[:,1])
roc_auc_1 = auc(false_positive_rate_1, true_positive_rate_1)
false_positive_rate_0, true_positive_rate_0, thresholds_0 = roc_curve(1-Y, classProb[:,0])
roc_auc_0 = auc(false_positive_rate_0, true_positive_rate_0)

plt.figure()
plt.plot(false_positive_rate_1, true_positive_rate_1, label='ROC curve for class {0} (area = {1:0.2f})'.format('1', roc_auc_1))
plt.plot(false_positive_rate_0, true_positive_rate_0, label='ROC curve for class {0} (area = {1:0.2f})'.format('0', roc_auc_0))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for MCI Converters')
plt.legend(loc="lower right")
plt.savefig('foo2.png')