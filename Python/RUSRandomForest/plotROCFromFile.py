__author__ = 'Sulantha'
import math
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from Python.RUSRandomForest import Config
import numpy

def getOptimalOparatingPoint(fpr, tpr, th):
    #distanceList = [math.sqrt(math.pow((fpr[i]), 2)+math.pow((tpr[i]-1), 2))for i in range(len(th))]
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
    plt.annotate('Sen : {0}, Spec: {1}, Th: {2}'.format(optY, 1-optX, optTH), xy=(optX, optY), xytext=(0.25, 0.75), arrowprops=dict(arrowstyle="->"), connectionstyle="angle3,angleA=0,angleB=-90")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_title)
    plt.legend(loc="lower right", prop={'size':'small', 'family':'sans-serif'})
    plt.tight_layout()
    plt.savefig(Config.figOutputPath + fileName)

if __name__ == '__main__':
    data = numpy.loadtxt(Config.figOutputPath+'CSF_ONLY_SensSpec.csv')
    tpr = data[:,0]
    fpr = 1-data[:,1]
    th = data[:,2]
    plotROC(fpr, tpr, th, 'CSF ONLY', 'ROC for MCI Converters (CSF)', 'CSF_ONLY_ROC.png')
