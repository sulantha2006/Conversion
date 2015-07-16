__author__ = 'Sulantha'
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

mci_df = pd.read_csv('../Classification_Table.csv', delimiter=',')
mci_df = mci_df.drop('ID', axis=1)
Y = mci_df.Conversion.values
mci_df = mci_df.drop('Conversion', axis=1)
X = mci_df.as_matrix()

rf = RandomForestClassifier(n_estimators=1000, n_jobs = -1, verbose = 0, class_weight = 'auto', min_samples_leaf=20)
cv = StratifiedKFold(Y, 10)
validated = cross_val_predict(rf, X, Y, cv=cv)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y, validated)
cm = confusion_matrix(Y, validated)
print(cm)
