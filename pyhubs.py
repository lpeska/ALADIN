# PyHubs module

import math
import operator
import numpy as np


# =============================================================================
# distance functions 
# =============================================================================

def manhattanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += abs(instance1[x] - instance2[x])
    return distance


def euclideanDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def qubedMinkowskiDistance(instance1, instance2):
    distance = 0
    for x in range(len(instance1)):
        distance += pow(abs((instance1[x] - instance2[x])), 3)
    return math.pow(distance, float(1) / float(3))


def chebyshevDistance(instance1, instance2):
    list = []
    for x in range(len(instance1)):
        z = abs((instance1[x] - instance2[x]))
        list.append(z)
    distance = max(list)
    return distance


def cosDistance(instance1, instance2):
    a = []
    b = []
    c = []
    for x in range(len(instance1)):
        a.append(instance1[x] * instance2[x])
        b.append((instance1[x] * instance1[x]))
        c.append((instance2[x] * instance2[x]))

    if sum(b) == 0 or sum(c) == 0:
        return 0
    return 1 - (float(sum(a)) / float(math.sqrt(sum(b)) * math.sqrt(sum(c))))


# -----------------------------------------------------------------------------
# Dynamic Time Warping, a distance measure that can be used to compare time 
# series. For more information about DTW, see:
#
# N. Tomasev, K. Buza, K. Marussy, P.B. Kis (2015): Hubness-aware 
# Classification, Instance Selection and Feature Construction: Survey and 
# Extensions to Time-Series, In: U. Stanczyk, L. Jain (eds.), Feature selection 
# for data and pattern recognition, Springer-Verlag
#
# K. Buza (2011): Fusion Methods for Time Series Classification, Peter Lang 
# Verlag
# -----------------------------------------------------------------------------

class DTW:
    'Dynamic Time Warping distance measure for time series'

    def __init__(self, r=10):
        self.r = r

    def calculate(self, ts_1, ts_2):
        if len(ts_1) < len(ts_2):
            ts1 = ts_2
            ts2 = ts_1
        else:
            ts1 = ts_1
            ts2 = ts_2
        last = 0
        m = np.zeros((len(ts1), len(ts2)), dtype=np.float)  # DTW matrix
        dl = len(ts1) - len(ts2)
        for j in range(len(ts2)):
            for i in range(max(0, j - self.r), min(j + self.r + dl, len(ts1))):
                dif = abs(ts1[i] - ts2[j])
                if (i == 0) and (j == 0):
                    minim = 0
                elif (j == 0):
                    minim = m[i - 1, j]
                elif (i == 0):
                    minim = m[i, j - 1]
                elif (i == j - self.r):
                    minim = min(m[i, j - 1], m[i - 1, j - 1])
                elif (i == j + self.r - 1 + dl):
                    minim = min(m[i - 1, j], m[i - 1, j - 1])
                else:
                    minim = min(m[i, j - 1], m[i - 1, j], m[i - 1, j - 1])
                m[i, j] = dif + minim
                last = m[i, j]
        return last


# =============================================================================
# Basic functions to evaluate classifiers and regressors
# =============================================================================

# -----------------------------------------------------------------------------
# Calculation of classification accuracy
# -----------------------------------------------------------------------------

def getAccuracy(testLabels, predictions):
    correct = 0
    for x in range(len(testLabels)):
        if testLabels[x] == predictions[x]:
            correct += 1
    return (float(correct) / float(len(testLabels)) * 100)


# -----------------------------------------------------------------------------
# Calculation of a confusion matrix
# -----------------------------------------------------------------------------

def confusion_matrix(test_labels, predicted_labels):
    uniq_test_labels = np.unique(test_labels)
    num_uniq_labels = len(uniq_test_labels)

    conf_matr = [[0 for x in range(len(uniq_test_labels))] for x in range(num_uniq_labels)]

    for i in range(len(test_labels)):
        for j in range(num_uniq_labels):
            if (predicted_labels[i] == uniq_test_labels[j]):
                for k in range(num_uniq_labels):
                    if (test_labels[i] == uniq_test_labels[k]):
                        conf_matr[k][j] += 1

    return np.matrix(conf_matr, dtype=np.float)


# -----------------------------------------------------------------------------
# Calculate regression error
# -----------------------------------------------------------------------------

def mean_absolute_error(testLabels, predictions):
    D_test = len(testLabels)
    dif = 0
    for x in range(D_test):
        a = predictions[x] - testLabels[x]
        dif += abs(a)
    mae = (float(1) / float(D_test)) * dif
    return mae


def root_mean_square_error(testLabels, predictions):
    D_test = len(testLabels)
    dif = 0
    for x in range(D_test):
        a = pow((predictions[x] - testLabels[x]), 2)
        dif += a
    rmse = math.sqrt((float(1) / float(D_test)) * dif)
    return rmse


def normalized_mean_absolute_error(testLabels, predictions):
    D_test = len(testLabels)
    dif = 0
    for x in range(D_test):
        a = float(predictions[x] - testLabels[x]) / float(abs(testLabels[x]))
        dif += abs(a)
    nmae = (float(1) / float(D_test)) * dif
    return nmae


# -----------------------------------------------------------------------------
# cv-fold crossvalidation for classifiers in the pyhubs module
# -----------------------------------------------------------------------------

def cross_validation(classifier, attrs, labels, cv=2, seed=42):
    cross_acc = []
    n = len(labels)

    if (cv > n):
        print(
            "\n[PyHubs-ERROR] Number of folds for the cross validation can not be more than the number of instances\n")
        return

    ids = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(ids)
    classifier.cv = cv

    for i in range(cv):

        test_ids = ids[(cv - i - 1) * n // cv: (cv - i) * n // cv]
        train_ids0 = []
        for j in range(cv):
            if (i != j):
                train_ids0 = np.concatenate((ids[(cv - j - 1) * n // cv: (cv - j) * n // cv], train_ids0))

        train_ids = train_ids0.tolist()

        classifier.fit(attrs[train_ids], labels[train_ids])
        p = classifier.predict(attrs[test_ids])
        accuracy = getAccuracy(labels[test_ids], p)
        cross_acc.append(accuracy)
    return (cross_acc, np.mean(cross_acc), np.std(cross_acc))


# -----------------------------------------------------------------------------
# cv-fold crossvalidation for regressors in the pyhubs module
# -----------------------------------------------------------------------------

def cross_validation_reg(reg, attrs, labels, cv=2, seed=42):
    array_mae = []
    array_rmse = []
    array_nmae = []
    n = len(labels)

    if (cv > n):
        print(
            "\n[PyHubs-ERROR] Number of folds for the cross validation can not be more than the number of instances\n")
        return

    ids = list(range(n))
    np.random.seed(seed)
    np.random.shuffle(ids)
    reg.cv = cv

    for i in range(cv):

        test_ids = ids[(cv - i - 1) * n // cv: (cv - i) * n // cv]
        train_ids0 = []
        for j in range(cv):
            if (i != j):
                train_ids0 = np.concatenate((ids[(cv - j - 1) * n // cv: (cv - j) * n // cv], train_ids0))

        train_ids = train_ids0.tolist()

        reg.fit(attrs[train_ids], labels[train_ids])
        p = reg.predict(attrs[test_ids])

        mae = mean_absolute_error(labels[test_ids], p)
        array_mae.append(mae)
        rmse = root_mean_square_error(labels[test_ids], p)
        array_rmse.append(rmse)
        nmae = normalized_mean_absolute_error(labels[test_ids], p)
        array_nmae.append(nmae)

    return (array_mae, np.mean(array_mae), array_rmse, np.mean(array_rmse), array_nmae, np.mean(array_nmae))


# =============================================================================
# Functions for semi-supervised training and classification 
# =============================================================================

# -----------------------------------------------------------------------------
# This function allows you to use the classifiers in PyHubs in self-training
# mode. The function assumes that the data is split into labeled and unlabeled 
# subsets and all the unlabeled data are available at training time. 
# The function returns: predicted labels of the unlabeled data.
# See the documentation and the examples for more details about how to use these
# functions.
# -----------------------------------------------------------------------------

def predict_with_self_training(classifier, labeled_data, labels, unlabeled_data, iterations=10):
    unlabeled_data = np.array(unlabeled_data, dtype=np.float)
    if iterations > len(unlabeled_data):
        iterations = len(unlabeled_data)

    ids = list(range(0, len(unlabeled_data)))  # these instances of the unlabeled data are used
    predicted_labels = np.full((len(unlabeled_data)), -1, dtype=np.float)

    for i in range(iterations):

        classifier.fit(labeled_data, labels)
        p = predict_with_certainty(classifier, unlabeled_data[ids])
        p = np.c_[p, range(len(p))]
        p = sorted(p, key=lambda row: row[1], reverse=True)

        if (i < iterations - 1):
            i0 = ids[int(p[0][
                             2])]  # The "original" ID (from the user's perspective of that instance which is classified in this iteration)
            predicted_labels[i0] = p[0][0]
            ids.remove(i0)

            labels = np.concatenate((labels, [p[0][0]]))
            training_data1 = np.concatenate((labeled_data, [unlabeled_data[p[0][2]]]))
        else:
            # we are in the last iteration!
            for j in range(len(p)):
                i0 = ids[int(p[j][
                                 2])]  # The "original" ID (from the user's perspective of that instance which is classified in this iteration)
                predicted_labels[i0] = p[j][0]

    return predicted_labels


# -----------------------------------------------------------------------------
# This function allows you to use the classifiers in PyHubs in self-training
# mode. The function assumes that the data is split into 3 subsets: (i) labeled 
# data, (ii) unlabeled data available at training time, (iii) test data that is
# not used when training the classifiers.
# The function returns:
# (1) predicted labels of the unlabeled data, i.e., subset (ii), and
# (2) predicted labels of the test data, i.e., subset (iii).   
# See the documentation and the examples for more details about how to use these
# functions.
# -----------------------------------------------------------------------------


def predict_with_self_training_unlab_and_test(classifier, labeled_data, labels, unlabeled_data, test_data,
                                              iterations=10):
    unlabeled_data = np.array(unlabeled_data, dtype=np.float)
    if iterations > len(unlabeled_data):
        iterations = len(unlabeled_data)

    ids = list(range(0, len(unlabeled_data)))  # these instances of the unlabeled data are used
    predicted_labels = np.full((len(unlabeled_data)), -1, dtype=np.float)

    for i in range(iterations):

        classifier.fit(labeled_data, labels)
        p = predict_with_certainty(classifier, unlabeled_data[ids])
        p = np.c_[p, range(len(p))]
        p = sorted(p, key=lambda row: row[1], reverse=True)

        if (i < iterations - 1):
            i0 = ids[int(p[0][
                             2])]  # The "original" ID (from the user's perspective of that instance which is classified in this iteration)
            predicted_labels[i0] = p[0][0]
            ids.remove(i0)

            labels = np.concatenate((labels, [p[0][0]]))
            training_data1 = np.concatenate((labeled_data, [unlabeled_data[p[0][2]]]))
        else:
            # we are in the last iteration!
            for j in range(len(p)):
                i0 = ids[int(p[j][
                                 2])]  # The "original" ID (from the user's perspective of that instance which is classified in this iteration)
                predicted_labels[i0] = p[j][0]

            p1 = classifier.predict(test_data)

    return (predicted_labels, p1)


# -----------------------------------------------------------------------------
# Subroutine used by the above two functions that implement semi-supervised
# classification. The user of the PyHubs module is not expected to interact
# with this function directly.
# -----------------------------------------------------------------------------

def predict_with_certainty(cls, test_data):
    num_test_instances = len(test_data)
    pred0 = cls.predict_proba(test_data)
    predicted_labels = np.full((len(test_data), 2), 0, dtype=np.float)

    for i in range(num_test_instances):
        predicted_labels[i][0] = cls.uniq_train_labels[np.argmax(pred0[i])]
        if (np.max(pred0[i]) > 0):
            if (cls.cert != 'standard'):
                n = 0  # how many times this test instance would appear as nearest neighbour of training instances
                for j in range(len(cls.train_data)):
                    distance = cls.metric(test_data[i], cls.train_data[j])
                    if (distance < cls.dist[j]):
                        n += 1
                predicted_labels[i][1] = n * float(np.max(pred0[i])) / float(
                    np.sum(pred0[i]))  # hubness-aware certainty score
            else:
                predicted_labels[i][1] = float(np.max(pred0[i])) / float(np.sum(pred0[i]))
        else:
            predicted_labels[i][1] = 0

    return predicted_labels


# =============================================================================
# Classifiers
# -----------------------------------------------------------------------------
# For more information about hubness-aware classifiers, see:
#
# N. Tomasev, K. Buza, K. Marussy, P.B. Kis (2015): Hubness-aware 
# Classification, Instance Selection and Feature Construction: Survey and 
# Extensions to Time-Series, In: U. Stanczyk, L. Jain (eds.), Feature 
# selection for data and pattern recognition, Springer-Verlag 
# -----------------------------------------------------------------------------
# For the usage of this implementation, see "example-supervised.py" and
# "example-semi-supervised.py"
# =============================================================================


# -----------------------------------------------
# KNN
# -----------------------------------------------

class KNN:
    'Supervised and semi-supervised  KNN'

    def __init__(self, k_pred, metric=euclideanDistance, train_weights='uniform', certainty_type='standard'):
        self.k_pred = k_pred
        self.metric = metric
        self.weights = train_weights
        self.cert = certainty_type

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

        if (self.cert != 'standard'):
            self.num_train_instances = len(train_data)
            self.dist = np.full((self.num_train_instances), 0, dtype=np.float)

            for i in range(self.num_train_instances):
                neighbors = []
                for j in range(self.num_train_instances):
                    if (i == j):
                        continue
                    neighbors.append((self.metric(train_data[i], train_data[j]), j))
                neighbors.sort(key=lambda x: x[0])

                self.dist[i] = neighbors[self.k_pred - 1][0]

    def predict(self, test_data):

        num_test_instances = len(test_data)
        predicted_labels = np.full((num_test_instances), 0, dtype=np.float)

        for i in range(num_test_instances):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])
            classVotes = {}

            for k in range(self.k_pred):
                id = neighbors[k][1]
                label = self.train_labels[id]
                if (self.weights != 'uniform'):
                    if label in classVotes:
                        classVotes[label] += self.weights[id]
                    else:
                        classVotes[label] = self.weights[id]
                else:
                    if label in classVotes:
                        classVotes[label] += 1
                    else:
                        classVotes[label] = 1

            sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
            predicted_labels[i] = sortedVotes[0][0]
        return predicted_labels

    def predict_proba(self, test_data):

        self.uniq_train_labels = np.unique(self.train_labels)
        num_classes = len(self.uniq_train_labels)
        predicted_labels = np.full((len(test_data), num_classes), 0, dtype=np.float)

        for i in range(len(test_data)):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            classVotes = {}
            sumVotes = 0
            for k in range(self.k_pred):
                id = neighbors[k][1]
                label = self.train_labels[id]
                if (self.weights != 'uniform'):
                    if label in classVotes:
                        classVotes[label] += self.weights[id]
                    else:
                        classVotes[label] = self.weights[id]
                    sumVotes += self.weights[id]
                else:
                    if label in classVotes:
                        classVotes[label] += 1
                    else:
                        classVotes[label] = 1
                    sumVotes += 1

            for c in range(num_classes):
                if c in classVotes:
                    predicted_labels[i][c] = float(classVotes[c]) / float(sumVotes)
                else:
                    predicted_labels[i][c] = 0

        return predicted_labels


# ----------------------------------------------------------------------
# HWKNN
# ----------------------------------------------------------------------		


class HWKNN:
    'Supervised and semi-supervised HWKNN'

    def __init__(self, k_fit, k_pred, metric=euclideanDistance, train_weights='uniform', certainty_type='standard'):
        self.k_fit = k_fit
        self.k_pred = k_pred
        self.metric = metric
        self.weights = train_weights
        self.cert = certainty_type

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.uniq_train_labels = np.unique(self.train_labels)
        self.num_train_instances = len(train_data)

        n = np.full((self.num_train_instances), 0, dtype=np.float)
        bn = np.full((self.num_train_instances), 0, dtype=np.float)
        gn = np.full((self.num_train_instances), 0, dtype=np.float)

        self.dist = np.full((self.num_train_instances), 0, dtype=np.float)

        for i in range(self.num_train_instances):
            neighbors = []
            for j in range(self.num_train_instances):
                if (i == j):
                    continue
                neighbors.append((self.metric(train_data[i], train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_fit):
                id = neighbors[k][1]
                label = train_labels[id]
                if (label == train_labels[i]):
                    gn[id] = gn[id] + 1
                else:
                    bn[id] = bn[id] + 1
                n[id] = n[id] + 1

            if (self.cert != 'standard'):
                self.dist[i] = neighbors[self.k_pred - 1][0]

        m = np.mean(bn)
        s = np.std(bn)
        self.w = np.full((self.num_train_instances), 1, dtype=np.float)
        for i in range(self.num_train_instances):
            if (s != 0):
                self.w[i] = math.exp(-float(bn[i] - m) / float(s))
            if (self.weights != 'uniform'):
                self.w[i] = self.w[i] * self.weights[i]

    def predict(self, test_data):

        num_test_instances = len(test_data)
        predicted_labels = np.full((num_test_instances), 0, dtype=np.float)

        for i in range(num_test_instances):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])
            classVotes = {}
            for k in range(self.k_pred):
                id = neighbors[k][1]
                label = self.train_labels[id]
                if label in classVotes:
                    classVotes[label] += self.w[id]
                else:
                    classVotes[label] = self.w[id]
            sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
            predicted_labels[i] = sortedVotes[0][0]
        return predicted_labels

    def predict_proba(self, test_data):

        num_classes = len(self.uniq_train_labels)
        predicted_labels = np.full((len(test_data), num_classes), 0, dtype=np.float)

        for i in range(len(test_data)):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            classVotes = {}
            sumVotes = 0
            for k in range(self.k_pred):
                id = neighbors[k][1]
                label = self.train_labels[id]
                if label in classVotes:
                    classVotes[label] += self.w[id]
                else:
                    classVotes[label] = self.w[id]
                sumVotes += self.w[id]

            for c in range(num_classes):
                if c in classVotes:
                    predicted_labels[i][c] = float(classVotes[c]) / float(sumVotes)
                else:
                    predicted_labels[i][c] = 0

        return predicted_labels


# ----------------------------------------------------------------------
# HFNN example
# ----------------------------------------------------------------------		

class HFNN:
    'Supervised and semi-supervised HFNN'

    def __init__(self, k_fit, k_pred, metric=euclideanDistance, train_weights='uniform', certainty_type='standard'):

        self.k_fit = k_fit
        self.k_pred = k_pred
        self.metric = metric
        self.weights = train_weights
        self.cert = certainty_type

    def fit(self, train_data, train_labels):

        self.majority_class = np.bincount(train_labels).argmax()
        self.uniq_train_labels = np.unique(train_labels)
        self.num_uniq_labels = len(self.uniq_train_labels)
        self.num_train_instances = len(train_data)
        self.train_data = train_data
        self.train_labels = train_labels

        nc = [[0 for x in range(self.num_uniq_labels)] for x in range(self.num_train_instances)]
        n = np.full((self.num_train_instances), 0, dtype=np.float)
        bn = np.full((self.num_train_instances), 0, dtype=np.float)
        gn = np.full((self.num_train_instances), 0, dtype=np.float)
        self.dist = np.full((self.num_train_instances), 0, dtype=np.float)

        for i in range(self.num_train_instances):
            neighbors = []
            for j in range(self.num_train_instances):
                if (i == j):
                    continue
                neighbors.append((self.metric(train_data[i], train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_fit):
                id = neighbors[k][1]
                label = train_labels[id]
                if (label == train_labels[i]):
                    gn[id] = gn[id] + 1
                else:
                    bn[id] = bn[id] + 1
                n[id] = n[id] + 1

                for l in range(self.num_uniq_labels):
                    if (train_labels[i] == self.uniq_train_labels[l]):
                        nc[id][l] = nc[id][l] + 1

            if (self.cert != 'standard'):
                self.dist[i] = neighbors[self.k_pred - 1][0]

        self.u = [[0 for x in range(len(self.uniq_train_labels))] for x in range(len(train_data))]

        for i in range(self.num_train_instances):
            for j in range(self.num_uniq_labels):
                if (n[i] == 0):
                    self.u[i][j] = 0
                else:
                    self.u[i][j] = float(nc[i][j]) / float(n[i])

    def predict(self, test_data):

        predicted_labels = np.full((len(test_data)), 0, dtype=np.float)

        for i in range(len(test_data)):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])
            classVotes = {}
            sumVotes = 0
            for k in range(self.k_pred):
                id = neighbors[k][1]
                for label_index in range(len(self.uniq_train_labels)):
                    if (self.weights != 'uniform'):
                        if label_index in classVotes:
                            classVotes[label_index] += self.weights[id] * self.u[id][label_index]
                        else:
                            classVotes[label_index] = self.weights[id] * self.u[id][label_index]
                        sumVotes += self.weights[id] * self.u[id][label_index]
                    else:
                        if label_index in classVotes:
                            classVotes[label_index] += self.u[id][label_index]
                        else:
                            classVotes[label_index] = self.u[id][label_index]
                        sumVotes += self.u[id][label_index]
            if (sumVotes != 0):
                sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
                predicted_labels[i] = self.uniq_train_labels[sortedVotes[0][0]]
            else:
                predicted_labels[i] = self.majority_class
        return predicted_labels

    def predict_proba(self, test_data):

        predicted_probabilities = np.full((len(test_data), len(self.uniq_train_labels)), 0, dtype=np.float)

        for i in range(len(test_data)):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            classVotes = {}
            sumVotes = 0
            for k in range(self.k_pred):
                id = neighbors[k][1]
                for label_index in range(len(self.uniq_train_labels)):
                    if (self.weights != 'uniform'):
                        if label_index in classVotes:
                            classVotes[label_index] += self.weights[id] * self.u[id][label_index]
                        else:
                            classVotes[label_index] = self.weights[id] * self.u[id][label_index]
                        sumVotes += self.weights[id] * self.u[id][label_index]
                    else:
                        if label_index in classVotes:
                            classVotes[label_index] += self.u[id][label_index]
                        else:
                            classVotes[label_index] = self.u[id][label_index]
                        sumVotes += self.u[id][label_index]

            if (sumVotes != 0):
                for c in range(len(self.uniq_train_labels)):
                    if c in classVotes:
                        predicted_probabilities[i][c] = float(classVotes[c]) / float(sumVotes)
                    else:
                        predicted_probabilities[i][c] = 0
            else:
                # default prediction if all the votes are
                for c in range(len(self.uniq_train_labels)):
                    predicted_probabilities[i][c] = float(1.0) / float(len(self.uniq_train_labels))

        return predicted_probabilities


# ----------------------------------------------------------------------
# NHBNN
# ----------------------------------------------------------------------

class NHBNN:
    'Supervised and semi-supervised NHBNN'

    def __init__(self, k_fit, k_pred, metric=euclideanDistance, pseudoinst=1, train_weights='uniform',
                 certainty_type='standard'):

        self.k_fit = k_fit
        self.k_pred = k_pred
        self.metric = metric
        self.weights = train_weights
        self.pseudoinst = pseudoinst
        self.cert = certainty_type

    def fit(self, train_data, train_labels):

        self.uniq_train_labels = np.unique(train_labels)
        self.num_uniq_labels = len(self.uniq_train_labels)
        self.num_train_instances = len(train_data)
        self.train_data = train_data
        self.train_labels = train_labels

        self.nc = [[0 for x in range(self.num_uniq_labels)] for x in range(self.num_train_instances)]
        self.n_labels = np.full((self.num_uniq_labels), 0, dtype=np.float)
        self.prob_labels = np.full((self.num_uniq_labels), 0, dtype=np.float)
        self.dist = np.full((self.num_train_instances), 0, dtype=np.float)

        for i in range(self.num_train_instances):
            neighbors = []
            for j in range(self.num_train_instances):
                if (i == j):
                    continue
                neighbors.append((self.metric(train_data[i], train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_fit):
                id = neighbors[k][1]
                label = train_labels[id]

                for l in range(self.num_uniq_labels):
                    if (train_labels[i] == self.uniq_train_labels[l]):
                        self.nc[id][l] += 1

            if (self.cert != 'standard'):
                self.dist[i] = neighbors[self.k_pred - 1][0]

        for m in range(self.num_uniq_labels):
            train_labels_list = np.array(train_labels, dtype=np.float).tolist()
            self.n_labels[m] = train_labels_list.count(m)
            self.prob_labels[m] = float(self.n_labels[m]) / float(len(train_labels_list))

    def predict(self, test_data):

        num_test_instances = len(test_data)
        prob_final = [[0 for x in range(len(self.uniq_train_labels))] for x in range(num_test_instances)]
        self.partial_probability = [[0 for x in range(len(self.uniq_train_labels))] for x in
                                    range(len(self.train_labels))]
        predicted_labels = np.full((num_test_instances), 0, dtype=np.float)

        for t in range(num_test_instances):
            for q in range(self.num_uniq_labels):
                prob_final[t][q] = self.prob_labels[q]

        for i in range(num_test_instances):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for h in range(self.num_uniq_labels):
                for z in range(self.k_pred):
                    id = neighbors[z][1]
                    self.partial_probability[id][h] = float(self.nc[id][h] + self.pseudoinst) / float(
                        self.n_labels[h] + self.pseudoinst * self.num_uniq_labels)
                    prob_final[i][h] *= self.partial_probability[id][h]
                    if (self.weights == 'uniform'):
                        prob_final[i][h] *= 1
                    else:
                        prob_final[i][h] *= self.weights[h]

            predicted_labels[i] = self.uniq_train_labels[np.argmax(prob_final[i])]

        return predicted_labels

    def predict_proba(self, test_data):

        num_test_instances = len(test_data)
        prob_final = [[0 for x in range(self.num_uniq_labels)] for x in range(num_test_instances)]
        self.partial_probability = [[0 for x in range(self.num_uniq_labels)] for x in range(len(self.train_labels))]
        predicted_probabilities = [[0 for x in range(self.num_uniq_labels)] for x in range(len(test_data))]

        for t in range(num_test_instances):
            for q in range(self.num_uniq_labels):
                prob_final[t][q] = self.prob_labels[q]

        for i in range(num_test_instances):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for h in range(self.num_uniq_labels):
                for z in range(self.k_pred):
                    id = neighbors[z][1]
                    self.partial_probability[id][h] = float(self.nc[id][h] + self.pseudoinst) / float(
                        self.n_labels[h] + self.pseudoinst * self.num_uniq_labels)
                    prob_final[i][h] *= self.partial_probability[id][h]
                    if (self.weights == 'uniform'):
                        prob_final[i][h] *= 1
                    else:
                        prob_final[i][h] *= self.weights[h]

            for we in range(self.num_uniq_labels):
                if sum(prob_final[i]) == 0:
                    predicted_probabilities[i][we] = 0
                else:
                    predicted_probabilities[i][we] = (float(prob_final[i][we]) / float(sum(prob_final[i])))

        return np.matrix(predicted_probabilities, dtype=np.float)


# ----------------------------------------------------------------------
# HIKNN
# ----------------------------------------------------------------------

class HIKNN:
    'Supervised and semi-supervised HIKNN'

    def __init__(self, k_fit, k_pred, metric=euclideanDistance, train_weights='uniform', certainty_type='standard'):
        self.k_fit = k_fit
        self.k_pred = k_pred
        self.metric = metric
        self.weights = train_weights
        self.cert = certainty_type

    def fit(self, train_data, train_labels):
        self.uniq_train_labels = np.unique(train_labels)
        self.num_uniq_labels = len(self.uniq_train_labels)
        self.num_train_instances = len(train_data)
        self.train_data = train_data
        self.train_labels = train_labels

        nc = [[0 for x in range(len(self.uniq_train_labels))] for x in
              range(self.num_train_instances)]  # N per each class array\
        self.n = np.full((self.num_train_instances), 0, dtype=np.float)
        self.u = [[0 for x in range(len(self.uniq_train_labels))] for x in range(self.num_train_instances)]
        self.dist = np.full((self.num_train_instances), 0, dtype=np.float)

        for i in range(self.num_train_instances):
            neighbors = []
            for j in range(self.num_train_instances):
                if (i == j):
                    continue
                neighbors.append((self.metric(train_data[i], train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_fit):
                id = neighbors[k][1]
                label = train_labels[id]
                self.n[id] = self.n[id] + 1

                for l in range(self.num_uniq_labels):
                    if (train_labels[i] == self.uniq_train_labels[l]):
                        nc[id][l] = nc[id][l] + 1

        if (self.cert != 'standard'):
            self.dist[i] = neighbors[self.k_pred - 1][0]

        for i in range(self.num_train_instances):
            for j in range(self.num_uniq_labels):
                if (self.n[i] == 0):
                    self.u[i][j] = 0
                else:
                    self.u[i][j] = float(nc[i][j]) / float(self.n[i])

    def predict(self, test_data):

        num_test_instances = len(test_data)
        p = np.full((self.num_train_instances), 0, dtype=np.float)
        I = np.full((self.num_train_instances), 0, dtype=np.float)
        alfa = np.full((self.num_train_instances), 0, dtype=np.float)
        beta = np.full((self.num_train_instances), 0, dtype=np.float)
        pk = [[0 for x in range(self.num_uniq_labels)] for x in range(self.k_fit)]
        uc = [[0 for x in range(self.num_uniq_labels)] for x in range(num_test_instances)]
        predicted_labels = np.full(num_test_instances, 0, dtype=np.float)

        for i in range(num_test_instances):
            neighbors = []
            for j in range(self.num_train_instances):
                if (self.n[j] > 1):
                    neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_pred):
                id = neighbors[k][1]
                p[id] = float(self.n[id]) / float(self.num_train_instances)

                I[id] = math.log(1.0 / p[id], self.n[id])
                beta[id] = float(I[id]) / float(math.log(self.num_train_instances, self.n[id]))

            for m in range(self.k_pred):
                id = neighbors[m][1]
                alfa[id] = float(I[id] - min(I)) / float(math.log(self.num_train_instances, self.n[id]) - min(I))

            for q in range(self.k_pred):
                for c in range(self.num_uniq_labels):
                    id = neighbors[q][1]
                    if (c == self.train_labels[id]):
                        pk[q][c] = alfa[id]
                    else:
                        pk[q][c] = 0
                    pk[q][c] += (1 - alfa[id]) * self.u[id][c]

            for w in range(self.num_uniq_labels):
                for t in range(self.k_pred):
                    id = neighbors[t][1]
                    uc[i][w] += beta[id] * pk[t][w]

                    if (self.weights == 'uniform'):
                        uc[i][w] *= 1
                    else:
                        uc[i][w] *= self.weights[id]
            predicted_labels[i] = self.uniq_train_labels[np.argmax(uc[i])]
        return predicted_labels

    def predict_proba(self, test_data):

        num_test_instances = len(test_data)
        p = np.full((self.num_train_instances), 0, dtype=np.float)
        I = np.full((self.num_train_instances), 0, dtype=np.float)
        alfa = np.full((self.num_train_instances), 0, dtype=np.float)
        beta = np.full((self.num_train_instances), 0, dtype=np.float)
        pk = [[0 for x in range(self.num_uniq_labels)] for x in range(self.k_fit)]
        uc = [[0 for x in range(self.num_uniq_labels)] for x in range(num_test_instances)]
        predicted_probabilities = [[0 for x in range(self.num_uniq_labels)] for x in range(len(test_data))]
        predicted_labels = np.full(num_test_instances, 0, dtype=np.float)

        for i in range(num_test_instances):
            neighbors = []
            for j in range(self.num_train_instances):
                if (self.n[j] > 1):
                    neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_pred):
                id = neighbors[k][1]
                p[id] = float(self.n[id]) / float(self.num_train_instances)

                I[id] = math.log(1.0 / p[id], self.n[id])
                beta[id] = float(I[id]) / float(math.log(self.num_train_instances, self.n[id]))

            for m in range(self.k_pred):
                id = neighbors[m][1]
                alfa[id] = float(I[id] - min(I)) / float(math.log(self.num_train_instances, self.n[id]) - min(I))

            for q in range(self.k_pred):
                for c in range(self.num_uniq_labels):
                    id = neighbors[q][1]
                    if (c == self.train_labels[id]):
                        pk[q][c] = alfa[id]
                    else:
                        pk[q][c] = 0
                    pk[q][c] += (1 - alfa[id]) * self.u[id][c]

            for w in range(self.num_uniq_labels):
                for t in range(self.k_pred):
                    id = neighbors[t][1]
                    uc[i][w] += beta[id] * pk[t][w]

                    if (self.weights == 'uniform'):
                        uc[i][w] *= 1
                    else:
                        uc[i][w] *= self.weights[id]

            for we in range(self.num_uniq_labels):
                if sum(uc[i]) == 0:
                    predicted_probabilities[i][we] = 0
                else:
                    predicted_probabilities[i][we] = (float(uc[i][we]) / float(sum(uc[i])))

        return np.matrix(predicted_probabilities, dtype=np.float)


# =============================================================================
# Regression approaches
# -----------------------------------------------------------------------------
# For the description of hubness-aware regressors, see:
#
# K. Buza, A. Nanopoulos, G. Nagy (2015): Nearest Neighbor Regression in the 
# Presence of Bad Hubs, Knowledge-Based Systems, Volume 86, pp. 250-260 
# -----------------------------------------------------------------------------
# For the usage of the implementated classes, see "example-reg.py"
# =============================================================================


# ----------------------------------------------------------------------
# KNNreg 
# ----------------------------------------------------------------------


class KNNreg:
    'KNN regression'

    def __init__(self, k_pred, metric=euclideanDistance):

        self.k_pred = k_pred
        self.metric = metric

    def fit(self, train_data, train_labels):

        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, test_data):

        num_test_data = len(test_data)
        predicted_labels = np.full(num_test_data, 0, dtype=np.float)

        for i in range(num_test_data):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            nominator = 0
            numerator = 0
            for k in range(self.k_pred):
                id = neighbors[k][1]
                label = self.train_labels[id]

                nominator += self.train_labels[id]
                numerator += 1

            predicted_labels[i] = float(nominator) / float(numerator)

        return predicted_labels


# ----------------------------------------------------------------------
# EWKNN 
# ----------------------------------------------------------------------


class EWKNN:
    'EWKNN'

    def __init__(self, k_fit, k_pred, metric=euclideanDistance):

        self.k_fit = k_fit
        self.k_pred = k_pred
        self.metric = metric

    def fit(self, train_data, train_labels):

        num_train_data = len(train_data)
        self.train_data = train_data
        self.train_labels = train_labels

        self.n = [[0 for xi in range(2)] for xi in range(num_train_data)]
        self.error = np.full((num_train_data), 0, dtype=np.float)

        for i in range(num_train_data):
            neighbors = []
            for j in range(num_train_data):
                if (i == j):
                    continue
                neighbors.append((self.metric(train_data[i], train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_fit):
                id = neighbors[k][1]
                self.n[id][0] += 1
                self.n[id][1] += abs(self.train_labels[i] - self.train_labels[id])

        for mi in range(num_train_data):
            if (self.n[mi][0] != 0):
                self.error[mi] = float(self.n[mi][1]) / float(self.n[mi][0])
            else:
                self.error[mi] = 0

        self.m = np.mean(self.error)
        self.s = np.std(self.error)

    def predict(self, test_data):

        num_test_data = len(test_data)
        predicted_labels = np.full(num_test_data, 0, dtype=np.float)

        for i in range(num_test_data):
            neighbors = []
            for j in range(len(self.train_data)):
                if (self.n[j][0] > 0):
                    neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            nominator = np.full(self.k_pred, 0, dtype=np.float)
            denominator = np.full(self.k_pred, 0, dtype=np.float)

            for k in range(self.k_pred):

                id = neighbors[k][1]
                label = self.train_labels[id]

                w = 1
                if (self.s != 0):
                    w = math.exp(- (float(self.error[id]) - float(self.m)) / float(self.s))

                nominator[k] = w * self.train_labels[id]
                denominator[k] = w

            sum_nominator = sum(nominator)
            sum_denominator = sum(denominator)
            predicted_labels[i] = float(sum_nominator) / float(sum_denominator)

        return predicted_labels


# ----------------------------------------------------------------------
# ECKNN 
# ----------------------------------------------------------------------

class ECKNN:
    'ECKNN'

    def __init__(self, k_fit, k_pred, metric=euclideanDistance):

        self.k_fit = k_fit
        self.k_pred = k_pred
        self.metric = metric

    def fit(self, train_data, train_labels):

        num_train_data = len(train_data)
        self.train_data = train_data
        self.train_labels = train_labels

        self.n = [[0 for xi in range(2)] for xi in range(num_train_data)]
        self.cor_train_labels = np.full((num_train_data), 0, dtype=np.float)

        for i in range(num_train_data):
            neighbors = []
            for j in range(num_train_data):
                if (i == j):
                    continue
                neighbors.append((self.metric(train_data[i], train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_fit):
                id = neighbors[k][1]
                self.n[id][0] += 1
                self.n[id][1] += self.train_labels[i]

        for mi in range(num_train_data):
            if (self.n[mi][0] != 0):
                self.cor_train_labels[mi] = float(self.n[mi][1]) / float(self.n[mi][0])
            else:
                self.cor_train_labels[mi] = train_labels[mi]

    def predict(self, test_data):

        num_test_data = len(test_data)
        predicted_labels = np.full(num_test_data, 0, dtype=np.float)

        for i in range(num_test_data):
            neighbors = []
            for j in range(len(self.train_data)):
                neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_pred):
                id = neighbors[k][1]
                label = self.cor_train_labels[id]

                predicted_labels[i] += (float(1) / float(self.k_pred) * float(self.cor_train_labels[id]))

        return predicted_labels


# ----------------------------------------------------------------------
# EWCKNN 
# ----------------------------------------------------------------------	      

class EWCKNN:
    'EWCKNN'

    def __init__(self, k_fit, k_pred, metric=euclideanDistance):

        self.k_fit = k_fit
        self.k_pred = k_pred
        self.metric = metric

    def fit(self, train_data, train_labels):

        num_train_data = len(train_data)
        self.train_data = train_data
        self.train_labels = train_labels

        self.n = [[0 for xi in range(3)] for xi in range(num_train_data)]
        self.error = np.full((num_train_data), 0, dtype=np.float)
        self.cor_train_labels = np.full((num_train_data), 0, dtype=np.float)

        for i in range(num_train_data):
            neighbors = []
            for j in range(num_train_data):
                if (i == j):
                    continue
                neighbors.append((self.metric(train_data[i], train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            for k in range(self.k_fit):
                id = neighbors[k][1]
                self.n[id][0] += 1
                self.n[id][1] += self.train_labels[i]
                self.n[id][2] += abs(self.train_labels[i] - self.train_labels[id])

        for mi in range(num_train_data):
            if (self.n[mi][0] != 0):
                self.cor_train_labels[mi] = float(self.n[mi][1]) / float(self.n[mi][0])
                self.error[mi] = float(self.n[mi][2] / self.n[mi][0])
            else:
                self.cor_train_labels[mi] = train_labels[mi]
                self.error[mi] = 0

        self.m = np.mean(self.error)
        self.s = np.std(self.error)

    def predict(self, test_data):

        num_test_data = len(test_data)
        predicted_labels = np.full(num_test_data, 0, dtype=np.float)

        for i in range(num_test_data):
            neighbors = []
            for j in range(len(self.train_data)):
                if (self.n[j][0] > 0):
                    neighbors.append((self.metric(test_data[i], self.train_data[j]), j))
            neighbors.sort(key=lambda x: x[0])

            nominator = np.full(self.k_pred, 0, dtype=np.float)
            denominator = np.full(self.k_pred, 0, dtype=np.float)

            for k in range(self.k_pred):

                id = neighbors[k][1]
                label = self.train_labels[id]

                w = 1
                if (self.s != 0):
                    w = math.exp(-1 * (float((self.error[id] - self.m)) / float(self.s)))

                nominator[k] = w * self.cor_train_labels[id]
                denominator[k] = w

            sum_nominator = sum(nominator)
            sum_denominator = sum(denominator)
            predicted_labels[i] = float(sum_nominator) / float(sum_denominator)

        return predicted_labels


# =============================================================================
# "SUCCESS" semi-supervised time-series classifiers
# -----------------------------------------------------------------------------
# For more information about the method, see:
#
# K. Marussy, K. Buza (2013): SUCCESS: A New Approach for Semi-Supervised 
# Classification of Time-Series, ICAISC, LNCS Vol. 7894, pp. 437-447, Springer. 
# -----------------------------------------------------------------------------
# For the usage of this implementation, see "example-success.py"
# =============================================================================

class SUCCESS:

    #	def __init__(self):

    def single_link(self, dmat, cannotlink):
        instance_cluster = np.array(range(len(dmat)), dtype=np.float)
        cluster_distances = np.array(dmat, dtype=np.float)
        max_plus_1 = np.max(cluster_distances) + 1
        for i in range(len(cluster_distances)):
            cluster_distances[i, i] = max_plus_1
        for i1 in cannotlink:
            for i2 in cannotlink:
                cluster_distances[i1, i2] = max_plus_1
        # print(str(cluster_distances[1,4]))
        num_clusters = len(dmat)
        while num_clusters > len(cannotlink):
            clusters_to_join = np.where(cluster_distances == np.min(cluster_distances))
            c1 = clusters_to_join[0][0]
            c2 = clusters_to_join[1][0]
            if (c2 < c1):
                tmp = c1
                c1 = c2
                c2 = tmp
            # print(str(c1)+","+str(c2)+" "+str(np.min(cluster_distances))+" "+str(instance_cluster))
            instances_in_c2 = np.where(instance_cluster == c2)
            for i in instances_in_c2:
                instance_cluster[i] = c1

            new_distance_vector = np.minimum(cluster_distances[:, c1], cluster_distances[:, c2])
            for pos in np.where((cluster_distances[:, c1] == max_plus_1) | (cluster_distances[:, c2] == max_plus_1)):
                new_distance_vector[pos] = max_plus_1
            cluster_distances[:, c1] = new_distance_vector
            cluster_distances[c1, :] = new_distance_vector
            cluster_distances[:, c2] = max_plus_1
            cluster_distances[c2, :] = max_plus_1
            cluster_distances[c1, c1] = max_plus_1

            num_clusters -= 1
        return instance_cluster

    def fit(self, labeled_train_time_series, unlabeled_train_time_series, train_labels):

        labeled_data_size = len(labeled_train_time_series)
        unlabeled_data_size = len(unlabeled_train_time_series)
        train_data_size = labeled_data_size + unlabeled_data_size

        dtw = DTW()
        dtw_distances_matrix = np.zeros((train_data_size, train_data_size), dtype=np.float)
        for i in range(train_data_size):
            for j in range(i + 1, train_data_size):
                if (i < labeled_data_size):
                    ts1 = labeled_train_time_series[i]
                else:
                    ts1 = unlabeled_train_time_series[i - labeled_data_size]

                if (j < labeled_data_size):
                    ts2 = labeled_train_time_series[j]
                else:
                    ts2 = unlabeled_train_time_series[j - labeled_data_size]
                dist = dtw.calculate(ts1, ts2)
                dtw_distances_matrix[i, j] = dist
                dtw_distances_matrix[j, i] = dist

        clustering = self.single_link(dtw_distances_matrix, list(range(labeled_data_size)))

        predicted_labels_unlabeled_train = np.zeros(unlabeled_data_size, dtype=np.float)
        cluster_ids = np.unique(clustering)
        for cid in cluster_ids:
            instances_in_a_cluster = np.where(clustering == cid)[0]
            label_of_cluster = int(train_labels[np.min(instances_in_a_cluster)])
            for i in instances_in_a_cluster:
                if i >= labeled_data_size:
                    predicted_labels_unlabeled_train[i - labeled_data_size] = label_of_cluster

        self.train_labels = train_labels
        self.predicted_labels_unlabeled_train = predicted_labels_unlabeled_train

        self.labeled_train_time_series = labeled_train_time_series
        self.unlabeled_train_time_series = unlabeled_train_time_series

        predicted_labels_unlabeled_train_int = []
        for p in predicted_labels_unlabeled_train:
            predicted_labels_unlabeled_train_int += [int(p)]

        return predicted_labels_unlabeled_train_int

    def predict(self, test_time_series):
        # Predictions for the test data
        predictions = []
        dtw = DTW()
        for i in range(len(test_time_series)):
            min_dist = dtw.calculate(self.labeled_train_time_series[0], test_time_series[i])
            pred_lab = self.train_labels[0]

            for j in range(1, len(self.labeled_train_time_series)):
                dist = dtw.calculate(self.labeled_train_time_series[j], test_time_series[i])
                if dist < min_dist:
                    min_dist = dist
                    pred_lab = self.train_labels[j]

            for j in range(len(self.unlabeled_train_time_series)):
                dist = dtw.calculate(self.unlabeled_train_time_series[j], test_time_series[i])
                if dist < min_dist:
                    min_dist = dist
                    pred_lab = self.predicted_labels_unlabeled_train[j]

            predictions += [int(pred_lab)]

        return predictions
