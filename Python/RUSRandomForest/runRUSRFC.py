__author__ = 'Sulantha'
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict, StratifiedKFold
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from Python.RUSRandomForest import RUSRandomForestClassifier

mci_df = pd.read_csv('../../Classification_Table.csv', delimiter=',')
mci_df = mci_df.drop('ID', axis=1)
Y = mci_df.Conversion.values
mci_df = mci_df.drop('Conversion', axis=1)
X = mci_df.as_matrix()

RUSRFC = RUSRandomForestClassifier.RUSRandomForestClassifier()
predClasses, classProb = RUSRFC.CVJungle(X, Y)
cm = confusion_matrix(Y, predClasses)
print(cm)