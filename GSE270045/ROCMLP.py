# Script that makes use of more advanced feature selection techniques
# by Alejandro Lopez, 2026

import copy
import numpy as np
import sys
import pandas as pd 

from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPClassifier

# used for normalization
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# used for cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

from numpy import interp
import matplotlib.pyplot as plt

# this is an incredibly useful function
from pandas import read_csv
from sklearn.utils import resample

randNumber=0
np.random.seed(randNumber)
numberOfFolds = 10



def loadDataset():
    index = 0
    dfData = read_csv("./data.csv", header=None, sep=',')
    dfLabels = read_csv("./labels.csv", header=None)

    return dfData.values, dfLabels.values.ravel()


def runFeatureReduce(numberOfFolds):

    classifierList = [
        #[LassoCV(), "LassoCV"],
        [MLPClassifier(random_state=randNumber), "MLPClassifier"]
    ]

    print("Loading dataset...")
    X, y = loadDataset()

    print(len(X))
    print(len(X[0]))
    print(len(y))

    labels = np.max(y) + 1

    skf = StratifiedKFold(n_splits=numberOfFolds, shuffle=True, random_state=randNumber)
    indexes = [(training, test) for training, test in skf.split(X, y)]

    classifierIndex = 0

    for originalClassifier, classifierName in classifierList:

        print("\nClassifier " + classifierName)

        classifierPerformance = []
        mccs = []

        fig1, ax1 = plt.subplots()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        foldIndex = 0

        for train_index, test_index in indexes:

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            classifier = copy.deepcopy(originalClassifier)
            classifier.fit(X_train, y_train)

            scoreTraining = classifier.score(X_train, y_train)
            scoreTest = classifier.score(X_test, y_test)

            y_new = classifier.predict(X_test)

            for i in range(0, len(y_new)):
                y_new[i] = round(y_new[i])

            # MCC per fold
            mcc = matthews_corrcoef(y_test, y_new)
            mccs.append(mcc)

            # ROC CURVE
            fpr, tpr, thresholds = roc_curve(y_test, y_new)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0

            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            ax1.plot(
                fpr, tpr,
                lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (foldIndex, roc_auc)
            )

            print("\ttraining: %.4f, test: %.4f, MCC: %.4f" %
                  (scoreTraining, scoreTest, mcc))

            classifierPerformance.append(scoreTest)
            foldIndex += 1

        classifierIndex += 1

        line = "%s \t %.4f \t %.4f \n" % (
            classifierName,
            np.mean(classifierPerformance),
            np.std(classifierPerformance)
        )

        print(line)

        # MCC summary
        mean_mcc = np.mean(mccs)
        std_mcc = np.std(mccs)
        print("MCC: %.4f ± %.4f" % (mean_mcc, std_mcc))

        # Chance line
        ax1.plot(
            [0, 1], [0, 1],
            linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8
        )

        # Mean ROC
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax1.plot(
            mean_fpr, mean_tpr,
            color='b',
            label = (r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc)
                + "\n"
                + r"MCC = %0.2f $\pm$ %0.2f" % (mean_mcc, std_mcc)),
            lw=2, alpha=.8
        )

        # Std shading
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        ax1.fill_between(
            mean_fpr, tprs_lower, tprs_upper,
            color='grey', alpha=.2,
            label=r'$\pm$ 1 std. dev.'
        )

        ax1.axis(xmin=-0.05, xmax=1.05)
        ax1.axis(ymin=-0.05, ymax=1.05)

        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')

        # Optional: MCC also in title
        ax1.set_title('ROC %s ' %
                      (classifierName))

        ax1.legend(loc="lower right")

        plt.savefig("%s.png" % (classifierName), dpi=300)

    return


if __name__ == "__main__":
    sys.exit(runFeatureReduce(numberOfFolds))