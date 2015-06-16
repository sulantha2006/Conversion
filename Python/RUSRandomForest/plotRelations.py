__author__ = 'sulantha'
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from Python.RUSRandomForest import Config

if __name__ == '__main__':
    class_data = pd.read_csv('../../Classification_Table_New.csv', delimiter=',')
    class_data = class_data.drop('ID', axis=1)
    plt.figure()
    #g = sns.PairGrid(class_data, hue='Conversion', palette='Set2')
    #g.map_diag(sns.kdeplot, lw=1)
    #g.map_upper(plt.scatter)
    #g.map_lower(plt.scatter)
    #g.add_legend()
    g = sns.pairplot(class_data, hue='Conversion', palette='Set2', diag_kind='kde', size=2.5)
    plt.savefig(Config.figOutputPath + 'Relations.png')



