from sklearn.mixture import GaussianMixture
from scipy.stats import beta

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

import math
import random
import matplotlib.pyplot as plt
import numpy as np
import collections


#code below from: https://git.cs.uni-paderborn.de/mhshaker/ida_paper114/-/tree/master?ref_type=heads
# @njit
def likelyhood(p, n, teta):
    # print(p,n)
    a = teta ** p
    b = (1 - teta) ** n
    c = (p / (n + p)) ** p
    d = (n / (n + p)) ** n

    return (a * b) / (c * d)

# @njit
def prob_pos(teta):
    return (2 * teta) - 1

# @njit
def prob_neg(teta):
    return 1 - (2 * teta)


def eyke_unc(pos, neg):
    sup_pos = 0
    sup_neg = 0
    for x in range(1, 100):
        x /= 100

        l = likelyhood(pos, neg, x)
        p_pos = prob_pos(x)
        min_pos = min(l, p_pos)

        if min_pos > sup_pos:
            sup_pos = min_pos

        p_neg = prob_neg(x)

        min_neg = min(l, p_neg)
        if min_neg > sup_neg:
            sup_neg = min_neg
    epistemic = min(sup_pos, sup_neg)
    aleatoric = 1 - max(sup_pos, sup_neg)
    total = epistemic + aleatoric

    return np.array([total, epistemic, aleatoric])

# @njit
def eyke_unc_vec(pos, neg):
    x_vals = np.linspace(0.01, 0.99, 99)  # Avoid 0 and 1 to prevent division by zero
    l_vals = likelyhood(pos, neg, x_vals)
    p_pos_vals = prob_pos(x_vals)
    p_neg_vals = prob_neg(x_vals)

    min_pos_vals = np.minimum(l_vals, p_pos_vals)
    min_neg_vals = np.minimum(l_vals, p_neg_vals)

    sup_pos = np.max(min_pos_vals)
    sup_neg = np.max(min_neg_vals)

    epistemic = min(sup_pos, sup_neg)
    aleatoric = 1 - max(sup_pos, sup_neg)
    total = epistemic + aleatoric

    return np.array([total, epistemic, aleatoric])


def relative_likelihood_uncertainty(counts):
    unc = np.zeros((counts.shape[0], counts.shape[1], 3))
    for i, x in enumerate(counts):
        for j, y in enumerate(x):
            unc[i, j] = eyke_unc_vec(y[1], y[0])
    unc_mean = np.mean(unc, axis=1)
    t = unc_mean[:, 0]
    e = unc_mean[:, 1]
    a = unc_mean[:, 2]
    return t, e, a

def relative_likelihood_uncertainty_old(counts):
    # print("------------------------")
    unc = np.zeros((counts.shape[0], counts.shape[1], 3))
    for i, x in enumerate(counts):
        for j, y in enumerate(x):
            res = eyke_unc(y[1], y[0])
            unc[i][j] = res
    unc = np.mean(unc, axis=1)
    t = unc[:, 0]
    e = unc[:, 1]
    a = unc[:, 2]
    return t, e, a

def entropy_uncertainty(probs): # three dimentianl array with d1 as datapoints, (d2) the rows as samples and (d3) the columns as probability for each class
    p = np.array(probs)
    entropy = -p*np.ma.log10(p)
    entropy = entropy.filled(0)
    a = np.sum(entropy, axis=1)
    a = np.sum(a, axis=1) / entropy.shape[1]
    p_m = np.mean(p, axis=1)
    total = -np.sum(p_m*np.ma.log10(p_m), axis=1)
    total = total.filled(0)
    e = total - a
    return total, e, a

def uncertainty_matrices(predictions):
    pred = predictions

    predictions_temp = [[[] for j in range(predictions.shape[0])] for i in range(predictions.shape[1])]
    prob_matrix = a = [[[] for j in range(predictions.shape[0])] for i in range(predictions.shape[1])]


    for model_index, model_prediction in enumerate(pred):
        for data_index in range(predictions.shape[1]):
            prob_matrix[data_index][model_index] = model_prediction[data_index]
            predictions_temp[data_index][model_index] = np.argmax(model_prediction[data_index])

    prediction_list = []
    for prob_predic_data in predictions_temp:
        counter = collections.Counter(prob_predic_data)
        temp = collections.Counter(counter)
        prediction_list.append(temp.most_common()[0][0])

    return np.array(prediction_list), np.array(prob_matrix)


class UncertaintyWrapperWithSpline(BaseEstimator, ClassifierMixin):
    def __init__(self, decision_threshold, num_classes, smoothing=0.01):
        self.decision_threshold = decision_threshold
        self.smoothing = smoothing
        self.num_classes = num_classes

        # Define the spline input-output pairs
        x = np.array([0, decision_threshold, 1])
        # y = np.array([np.log(smoothing), np.log(0.5), np.log(1 - smoothing)])
        y = np.array([np.log(smoothing), np.log(decision_threshold), np.log(smoothing)])
        # # Create the spline
        spline = CubicSpline(x, y)

        # # Transform function
        self.transform_function = lambda prob_pred: (np.exp(spline(prob_pred)) - 0.5) / (0.5 - smoothing) * 0.5 + 0.5

    def fit(self, X, y):
        # No fitting needed
        return self

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        # Transform probabilities
        prob_pred = (1 / np.log(self.num_classes)) * X
        p0 = self.transform_function(prob_pred)
        # print(p0)
        # p1 = np.ones(p0.shape[0]) - (p0 < 0.5)*p0 - (p0 >= 0.5)
        # Ensure probabilities sum to 1
        p0 = np.clip(p0, 0, .99)  # Probability for class 0
        p1 = 1 - p0  # Probability for class 1
        probabilities = np.stack([p0, p1], axis=1)

        return probabilities


from scipy.special import expit


def sigmoid(x):
    return 1 / (1 + np.exp(-x * 20))


class UncertaintyWrapperWithSigmoid(BaseEstimator, ClassifierMixin):
    def __init__(self, decision_threshold, smoothing=0.01):
        self.decision_threshold = decision_threshold
        self.smoothing = smoothing

        # Define the spline input-output pairs
        # x = np.array([0, decision_threshold, 1])
        # # y = np.array([np.log(smoothing), np.log(0.5), np.log(1 - smoothing)])
        # y = np.array([np.log(smoothing), np.log(decision_threshold), np.log(smoothing)])
        # # Create the spline
        # spline = #CubicSpline(x, y)

        # # Transform function
        self.transform_function = lambda prob_pred: (np.exp(self.spline(prob_pred)) - 0.5) / (
                    0.5 - smoothing) * 0.5 + 0.5

    def fit_spline(self, decision_threshold):
        # Define the spline input-output pairs
        x = np.array([0, decision_threshold, 1])
        # y = np.array([np.log(smoothing), np.log(0.5), np.log(1 - smoothing)])
        y = np.array([np.log(self.smoothing), np.log(decision_threshold), np.log(self.smoothing)])
        # # Create the spline
        self.spline = CubicSpline(x, y)
        # return self.spline

    def fit(self, X, y):
        # No fitting needed
        return self

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        # Transform probabilities
        prob_pred = X - self.decision_threshold
        # threshold = np.mean(prob_pred) + np.std(prob_pred)
        # self.fit_spline(threshold)
        p0 = sigmoid(prob_pred)
        # print(p0)
        # p1 = np.ones(p0.shape[0]) - (p0 < 0.5)*p0 - (p0 >= 0.5)
        # Ensure probabilities sum to 1
        # p0 = (1/np.log(2))*X#np.c/lip(p, 0, 1)  # Probability for class 0
        p1 = 1 - p0  # Probability for class 1
        probabilities = np.stack([p0, p1], axis=1)

        return probabilities

#threshold = crossover_original
def assign_uncertainty_label(value, threshold):
    if value < threshold :
        return 1  # Low uncertainty
    else: #(mean - std) <= value <= (mean + std):
        return 0
def fit_beta_mixture(data, n_components=2, num_points=1000):
    # Gaussian Mixture Model to estimate clusters
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    data_reshaped = data.reshape(-1, 1)
    gmm.fit(data_reshaped)
    labels = gmm.predict(data_reshaped)

    # Fit Beta distributions to each cluster
    betas = []
    x = np.linspace(0, 1, num_points)
    for label in range(n_components):
        cluster_data = data[labels == label]
        alpha, beta_, _, _ = beta.fit(cluster_data, floc=0, fscale=1)
        betas.append((alpha, beta_))

    return betas, labels


# Define the Beta PDFs
def beta_pdf(x, alpha, beta_):
    return beta.pdf(x, alpha, beta_, loc=0, scale=1)


def find_root(alpha1, beta1, alpha2, beta2, num_points=1000):
    def pdf1(x):
        return beta.pdf(x, alpha1, beta1)

    def pdf2(x):
        return beta.pdf(x, alpha2, beta2)

    # Define the difference between the two PDFs
    def diff(x):
        return pdf1(x) - pdf2(x)

    intersection = root_scalar(diff, bracket=[0.001, .999], method='brentq')

    return intersection

# Find the crossover point
def find_crossover(alpha1, beta1, alpha2, beta2, num_points=1000):
    x = np.linspace(0, 1, num_points)
    pdf1 = beta_pdf(x, alpha1, beta1)
    pdf2 = beta_pdf(x, alpha2, beta2)
    diff = np.abs(pdf1 - pdf2)

    crossover_idx = np.argmin(diff)
    if diff[crossover_idx] > 1e-6:
        return x[crossover_idx]
    else:
        crossover_idx = np.argsort(diff)[1]
        return x[crossover_idx]

