__author__ = 'Sulantha'
import math
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from Python.RUSRandomForest import Config
import numpy

def getOptimalOparatingPoint(fpr, tpr, th):
    distanceList = numpy.sqrt(numpy.power(fpr, 2) + numpy.power(tpr-1, 2))
    minIdx= numpy.argmin(distanceList)
    return tpr[minIdx], fpr[minIdx], th[minIdx]

def plotROC(fpr, tpr, th, legend_title, plot_title, fileName):
    plt.figure()
    plt.plot(fpr, tpr, 'r', label='ROC curve {0} (area = {1:0.2f})'.format(legend_title, auc(fpr, tpr)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    optY, optX, optTH = getOptimalOparatingPoint(fpr, tpr, th)
    plt.plot(optX, optY, 'o')
    plt.annotate('Sen : {0}\nSpec: {1}\nTh: {2}'.format(optY, 1-optX, optTH), xy=(optX, optY), xytext=(0.25, 0.75), arrowprops=dict(arrowstyle='fancy', fc='0.6', ec='none', shrinkB=5, connectionstyle="angle3,angleA=0,angleB=-45"))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_title)
    plt.legend(loc="lower right", prop={'size':'small', 'family':'sans'})
    plt.tight_layout()
    plt.savefig(Config.figOutputPath + fileName)

if __name__ == '__main__':
    data = numpy.loadtxt(Config.figOutputPath+'CSF_ONLY_SensSpec.csv', delimiter=',')
    plotROC(1-data[:, 0], data[:, 1], data[:, 2], 'CSF ONLY', 'ROC for MCI Converters (CSF)', 'CSF_ONLY_ROC.png')

    data = numpy.loadtxt(Config.figOutputPath+'AV45_ONLY_SensSpec.csv', delimiter=',')
    plotROC(1-data[:, 0], data[:, 1], data[:, 2], 'AV45 ONLY', 'ROC for MCI Converters (AV45)', 'AV45_ONLY_ROC.png')

    data = numpy.loadtxt(Config.figOutputPath+'FDG_ONLY_SensSpec.csv', delimiter=',')
    plotROC(1-data[:, 0], data[:, 1], data[:, 2], 'FDG ONLY', 'ROC for MCI Converters (FDG)', 'FDG_ONLY_ROC.png')

    data = numpy.loadtxt(Config.figOutputPath+'CSF_AV45_SensSpec.csv', delimiter=',')
    plotROC(1-data[:, 0], data[:, 1], data[:, 2], 'CSF AV45', 'ROC for MCI Converters (CSF & AV45)', 'CSF_AV45_ROC.png')

    data = numpy.loadtxt(Config.figOutputPath+'CSF_FDG_SensSpec.csv', delimiter=',')
    plotROC(1-data[:, 0], data[:, 1], data[:, 2], 'CSF FDG', 'ROC for MCI Converters (CSF & FDG)', 'CSF_FDG_ROC.png')

    data = numpy.loadtxt(Config.figOutputPath+'ALL_SensSpec.csv', delimiter=',')
    plotROC(1-data[:, 0], data[:, 1], data[:, 2], 'ALL', 'ROC for MCI Converters (ALL)', 'ALL_ROC.png')
