import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset

import urllib.request
from fastai.vision.all import *
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.interpolate import CubicSpline
import glob
from torchvision import transforms
import random
import torch.optim as optim
from sklearn.metrics import f1_score
from xplique.concepts.craft import BaseCraft, DisplayImportancesOrder, Factorization, Sensitivity
from xplique.attributions.global_sensitivity_analysis.sobol_estimators import SobolEstimator
from scipy.stats import entropy
from sklearn.decomposition import non_negative_factorization


def activation_transform(inputs,basis, patches=False, labels=None, activations=None, n_patches=16):
    """
    Transforms the input images into an (N, 320) representation where N is the number of images.

    Parameters:
    - inputs: Input images or data to be transformed.
    - patches: Whether to use patches (if needed for some other functionality).
    - labels: Optional labels for the inputs.
    - activations: Optional pre-computed activations. If None, activations are computed.
    - drift_basis: Predefined basis for NMF.
    - n_patches: Number of patches per image (default is 16).

    Returns:
    - transformed_data: Transformed dataset with shape (N, 320).
    """

    # Step 1: Extract latent activations using drift_craft
    A = inputs  # drift_craft._latent_predict(inputs)  # Assuming A.shape = (N, H, W, D) where D is the activation dimension
    # print(A.shape)
    # Step 2: Reshape activations to 2D (flatten the spatial dimensions)
    original_shape = A.shape[:-1]  # Keep original shape to reconstruct later
    re_activations = np.reshape(A, (-1, A.shape[-1]))
    # print(re_activations.shape)# Flatten to (N * H * W, D)
    # print(re_activations.shape)
    # Step 3: Apply Non-negative Matrix Factorization (NMF) to reduce dimensionality

    embedding, basis, n_iter = non_negative_factorization(np.array(re_activations),
                                                          n_components=len(basis),
                                                          init='custom',
                                                          update_H=False, solver='mu', H=basis)

    embedding = np.reshape(embedding, (*original_shape, embedding.shape[-1]))

    return embedding

from xplique.attributions.global_sensitivity_analysis import HaltonSequenceRS, JansenEstimator
# def estimate_importance_sigmoid(craft_instance, pred_fn, basis, inputs, nb_design: int = 32, verbose: bool = False):
#     """
#     Estimates the importance of each concept for all the classes of interest.
#
#     Parameters
#     ----------
#     nb_design
#         The number of design to use for the importance estimation. Default is 32.
#     verbose
#         If True, then print the current class CRAFT is estimating importances for,
#         otherwise no textual output will be printed.
#     """
#     y_preds, _ = pred_fn(inputs)
#     # print(y_preds)
#
#     global_importance = []
#     for class_of_interest in [0, 1]:
#         filtered_indices = np.where(y_preds == class_of_interest)
#         class_inputs = inputs[filtered_indices]
#         # print(class_inputs.shape)
#         importances = estimate_importance_helper_sigmoid(craft_instance, pred_fn,basis,inputs=class_inputs,
#                                                          # activations = class_activations,
#                                                          class_of_interest=class_of_interest,
#                                                          nb_design=nb_design,
#                                                          compute_class_importance=True)
#         global_importance.append(importances)
#
#     return global_importance
#
#
# def estimate_importance_helper_sigmoid(craft_instance, pred_fn,basis,  inputs: np.ndarray = None, class_of_interest: int = None,
#                                        nb_design: int = 32, compute_class_importance: bool = False) -> np.ndarray:
#     """
#     Estimates the importance of each concept for a given class, either globally
#     on the whole dataset provided in the fit() method (in this case, inputs shall
#     be set to None), or locally on a specific input image.
#
#     Parameters
#     ----------
#     inputs : numpy array or Tensor
#         The input data on which to compute the importances.
#         If None, then the inputs provided in the fit() method
#         will be used (global importance of the whole dataset).
#         Default is None.
#     nb_design
#         The number of design to use for the importance estimation. Default is 32.
#
#     Returns
#     -------
#     importances
#         The Sobol total index (importance score) for each concept.
#
#     """
#
#     coeffs_u = activation_transform(inputs.permute((0, 2, 3, 1)).cpu().numpy(),basis)
#     # print(coeffs_u.shape)
#     # coeffs_u = sigmoidunc_craft.transform(inputs)
#
#     masks = HaltonSequenceRS()(len(basis), nb_design=nb_design)
#     estimator = JansenEstimator()
#     importances = []
#
#     if len(coeffs_u.shape) == 4:
#         # for coeff in coeffs_u:
#         # apply a re-parameterization trick and use mask on all localization for a given
#         # concept id to estimate sobol indices
#         for coeff in coeffs_u:
#             u_perturbated = masks[:, None, None, :] * coeff[None, :]
#
#             a_perturbated = np.reshape(u_perturbated,
#                                        (-1, coeff.shape[-1])) @ basis
#             # print("a_perturbed", a_perturbated.shape)
#             a_perturbated = np.reshape(a_perturbated,
#                                        (len(masks), coeffs_u.shape[1], coeffs_u.shape[2], -1))
#             # print("a_perturbed-re", torch.from_numpy(a_perturbated).shape)
#
#             # a_perturbated: (N, H, W, C)
#             _, y_pred = pred_fn(torch.from_numpy(a_perturbated).permute((0, 3, 1, 2)))
#             # print("preds",y_pred.shape)
#
#             y_pred = y_pred[:, class_of_interest]
#
#             stis = estimator(masks, y_pred, nb_design)
#
#             importances.append(stis)
#
#     importances = np.mean(importances, 0)
#
#     # # Save the results of the computation if working on the whole dataset
#     # if compute_class_importance:
#     #     most_important_concepts = np.argsort(importances)[::-1]
#     #     craft_instance.sensitivities[class_of_interest] = Sensitivity(importances, most_important_concepts,
#     #                                                                     cmaps=plt.get_cmap(
#     #                                                                         'tab20b').colors + plt.get_cmap(
#     #                                                                         'Set3').colors)
#     if compute_class_importance:
#         most_important_concepts = np.argsort(importances)[::-1]
#         craft_instance.sensitivities[class_of_interest] = Sensitivity(importances, most_important_concepts,
#                                                                       cmaps=plt.get_cmap('tab20b').colors +
#                                                                             plt.get_cmap('tab20c').colors +
#                                                                             plt.get_cmap('tab20').colors +
#                                                                             plt.get_cmap('Set3').colors +
#                                                                       plt.get_cmap('Set1').colors +
#                                                                       plt.get_cmap('Set2').colors +
#                                                                       plt.get_cmap('tab10').colors +
#                                                                       plt.get_cmap('Pastel1').colors +
#                                                                       plt.get_cmap('Pastel2').colors)
#
#     return importances


def estimate_importance_sigmoid(craft_instance, pred_fn, basis, inputs, nb_design: int = 32, verbose: bool = False):
    """
    Estimates the importance of each concept for all the classes of interest.

    Parameters
    ----------
    nb_design
        The number of design to use for the importance estimation. Default is 32.
    verbose
        If True, then print the current class CRAFT is estimating importances for,
        otherwise no textual output will be printed.
    """
    y_preds, _ = pred_fn(inputs)
    # print(y_preds)
    local_importances = []
    global_importance = []
    for class_of_interest in [0, 1]:
        filtered_indices = np.where(y_preds == class_of_interest)
        class_inputs = inputs[filtered_indices]
        # print(class_inputs.shape)
        g_importances, l_importances = estimate_importance_helper_sigmoid(craft_instance, pred_fn, basis, inputs=class_inputs,
                                                         # activations = class_activations,
                                                         class_of_interest=class_of_interest,
                                                         nb_design=nb_design,
                                                         compute_class_importance=True)

        local_importances.extend(l_importances)
        global_importance.append(g_importances)

    return np.array(global_importance), np.array(local_importances)


def estimate_importance_helper_sigmoid(craft_instance, pred_fn, basis, inputs: np.ndarray = None,
                                       class_of_interest: int = None,
                                       nb_design: int = 32, compute_class_importance: bool = False) -> np.ndarray:
    """
    Estimates the importance of each concept for a given class, either globally
    on the whole dataset provided in the fit() method (in this case, inputs shall
    be set to None), or locally on a specific input image.

    Parameters
    ----------
    inputs : numpy array or Tensor
        The input data on which to compute the importances.
        If None, then the inputs provided in the fit() method
        will be used (global importance of the whole dataset).
        Default is None.
    nb_design
        The number of design to use for the importance estimation. Default is 32.

    Returns
    -------
    importances
        The Sobol total index (importance score) for each concept.

    """

    coeffs_u = activation_transform(inputs.permute((0, 2, 3, 1)).cpu().numpy(), basis)
    # print(coeffs_u.shape)
    # coeffs_u = sigmoidunc_craft.transform(inputs)

    masks = HaltonSequenceRS()(len(basis), nb_design=nb_design)
    estimator = JansenEstimator()
    importances = []

    if len(coeffs_u.shape) == 4:
        # for coeff in coeffs_u:
        # apply a re-parameterization trick and use mask on all localization for a given
        # concept id to estimate sobol indices
        for coeff in coeffs_u:
            u_perturbated = masks[:, None, None, :] * coeff[None, :]

            a_perturbated = np.reshape(u_perturbated,
                                       (-1, coeff.shape[-1])) @ basis
            # print("a_perturbed", a_perturbated.shape)
            a_perturbated = np.reshape(a_perturbated,
                                       (len(masks), coeffs_u.shape[1], coeffs_u.shape[2], -1))
            # print("a_perturbed-re", torch.from_numpy(a_perturbated).shape)

            # a_perturbated: (N, H, W, C)
            _, y_pred = pred_fn(torch.from_numpy(a_perturbated).permute((0, 3, 1, 2)))
            # print("preds",y_pred.shape)

            y_pred = y_pred[:, class_of_interest]

            stis = estimator(masks, y_pred, nb_design)

            importances.append(stis)

    g_importances = np.mean(importances, 0)

    # # Save the results of the computation if working on the whole dataset
    if compute_class_importance:
        most_important_concepts = np.argsort(g_importances)[::-1]
        craft_instance.sensitivities[class_of_interest] = Sensitivity(g_importances, most_important_concepts,
                                                                      cmaps=plt.get_cmap('tab20b').colors
                                                                            + plt.get_cmap('Set3').colors +
                                                                            plt.get_cmap('tab20c').colors +
                                                                            plt.get_cmap('tab10').colors +
                                                                            plt.get_cmap('tab20').colors +
                                                                            plt.get_cmap('Set1').colors +
                                                                            plt.get_cmap('Set2').colors
                                                                      )

    return g_importances, np.array(importances)




def local_imp_concepts_weighted_hard(craft_instance, importances, num, preds):
    # This gets three most important local concepts and adds up there importance globally with respect to each class and then finds the max
    image_preds = []
    # image_raw =[]
    for i, image_imp in enumerate(importances):
        max_local_3 = np.argsort(image_imp)[::-1][:num]

        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in max_local_3:
                # print(top_3)e
                argument = craft_instance.sensitivities[label].importances[top_3]
                arguments.append(argument)

            label_imp.append(np.sum(arguments))
        # print(label_imp)
        if np.argmax(label_imp) == 1:
            image_preds.append(preds[i][1] * np.argmax(label_imp))
        else:
            image_preds.append(preds[i][0] * -1)

    return image_preds


def local_imp_concepts_weighted_soft(craft_instance, importances, num, preds):
    # This gets three most important local concepts and adds up there importance globally with respect to each class and then finds the max
    image_preds = []
    # image_raw =[]
    for i, image_imp in enumerate(importances):
        max_local_3 = np.argsort(image_imp)[::-1][:num]

        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in max_local_3:
                # print(top_3)e
                argument = craft_instance.sensitivities[label].importances[top_3]
                arguments.append(argument)

            label_imp.append(np.sum(arguments))

        if np.argmax(label_imp) == 1:
            image_preds.append(preds[i][1] * label_imp[np.argmax(label_imp)])
        else:
            image_preds.append(preds[i][0] * -1 * label_imp[np.argmax(label_imp)])

    return image_preds


def global_imp_concepts_weighted_hard(craft_instance, importances, num, preds):
    # This method takes the top 3 global concepts from each class and sums their local importance. This method gets a 75% accuracy
    image_preds = []
    # image_raw =[]
    for i, image_imp in enumerate(importances):
        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in craft_instance.sensitivities[label].most_important_concepts[:num]:
                local = image_imp[top_3]
                arguments.append(local)
            label_imp.append(np.sum(arguments))
        if np.argmax(label_imp) == 1:
            image_preds.append(preds[i][1] * np.argmax(label_imp))
        else:
            image_preds.append(preds[i][0] * -1)

    return image_preds


def global_imp_concepts_weighted_soft(craft_instance, importances, num, preds):
    # This method takes the top 3 global concepts from each class and sums their local importance. This method gets a 75% accuracy
    image_preds = []
    # image_raw =[]
    for i, image_imp in enumerate(importances):
        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in craft_instance.sensitivities[label].most_important_concepts[:num]:
                local = image_imp[top_3]
                arguments.append(local)
            label_imp.append(np.sum(arguments))
        if np.argmax(label_imp) == 1:
            image_preds.append(preds[i][1] * label_imp[np.argmax(label_imp)])
        else:
            image_preds.append(preds[i][0] * -1 * label_imp[np.argmax(label_imp)])

    return image_preds


def local_imp_concepts_raw_hard(craft_instance, importances, num, preds):
    # This gets three most important local concepts and adds up there importance globally with respect to each class and then finds the max
    image_preds = []
    # image_raw =[]
    for i, image_imp in enumerate(importances):
        max_local_3 = np.argsort(image_imp)[::-1][:num]

        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in max_local_3:
                # print(top_3)e
                argument = craft_instance.sensitivities[label].importances[top_3]
                arguments.append(argument)

            label_imp.append(np.sum(arguments))
        # print(label_imp)
        if np.argmax(label_imp) == 1:
            image_preds.append(np.argmax(label_imp))
        else:
            image_preds.append(-1)

    return image_preds


def local_imp_concepts_raw_soft(craft_instance, importances, num, preds):
    # This gets three most important local concepts and adds up there importance globally with respect to each class and then finds the max
    image_preds = []
    # image_raw =[]
    for i, image_imp in enumerate(importances):
        max_local_3 = np.argsort(image_imp)[::-1][:num]

        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in max_local_3:
                # print(top_3)e
                argument = craft_instance.sensitivities[label].importances[top_3]
                arguments.append(argument)

            label_imp.append(np.sum(arguments))

        if np.argmax(label_imp) == 1:
            image_preds.append(label_imp[np.argmax(label_imp)])
        else:
            image_preds.append(-1 * label_imp[np.argmax(label_imp)])

    return image_preds


def global_imp_concepts_raw_hard(craft_instance, importances, num, preds):
    # This method takes the top 3 global concepts from each class and sums their local importance. This method gets a 75% accuracy
    image_preds = []
    # image_raw =[]
    for i, image_imp in enumerate(importances):
        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in craft_instance.sensitivities[label].most_important_concepts[:num]:
                local = image_imp[top_3]
                arguments.append(local)
            label_imp.append(np.sum(arguments))
        if np.argmax(label_imp) == 1:
            image_preds.append(np.argmax(label_imp))
        else:
            image_preds.append(-1 * np.argmax(label_imp))

    return image_preds


def global_imp_concepts_raw_soft(craft_instance, importances, num, preds):
    # This method takes the top 3 global concepts from each class and sums their local importance. This method gets a 75% accuracy
    image_preds = []
    # image_raw =[]
    for i, image_imp in enumerate(importances):
        label_imp = []
        for label in [0, 1]:
            arguments = []
            for top_3 in craft_instance.sensitivities[label].most_important_concepts[:num]:
                local = image_imp[top_3]
                arguments.append(local)
            label_imp.append(np.sum(arguments))
        if np.argmax(label_imp) == 1:
            image_preds.append(label_imp[np.argmax(label_imp)])
        else:
            image_preds.append(-1 * label_imp[np.argmax(label_imp)])

    return image_preds

class NewsClassifierNN(nn.Module):
    def __init__(self, out_features=2):
        super(NewsClassifierNN, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.relu(self.fc4(x))
        return x


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
        probabilities = np.stack([p1, p0], axis=1)

        return probabilities


from scipy.special import expit

from skimage.filters import threshold_otsu
from DriftLocalization.j_helper_functions import fit_beta_mixture, find_root
def get_threshold(t):
    t_min, t_max = np.min(t), np.max(t)
    # Add a small epsilon to ensure values are strictly within (0, 1)
    epsilon = 1e-6
    t_normalized = (t - t_min) / (t_max - t_min)
    t_normalized = np.clip(t_normalized, epsilon, 1 - epsilon)

    # Fit the Beta mixture and calculate crossover point
    try:
        betas, mix_labels = fit_beta_mixture(t_normalized)
        (alpha_low, beta_low), (alpha_high, beta_high) = betas

        crossover_normalized = find_root(alpha_low, beta_low, alpha_high, beta_high)

        # Transform the crossover point back to the original scale
        crossover_original = crossover_normalized.root #* (t_max - t_min) + t_min
        # threshold = threshold_otsu(t)
        threshold = crossover_original
        return threshold, (alpha_low, beta_low, alpha_high, beta_high), t_normalized
    except:
        threshold = threshold_otsu(t_normalized)
        return threshold, None, t_normalized
def sigmoid(x):
    return 1 / (1 + np.exp(-x*5))
class UncertaintyWrapperWithSigmoid(BaseEstimator, ClassifierMixin):
    def __init__(self, decision_threshold, smoothing=0.01):
        self.decision_threshold = decision_threshold
        self.smoothing = smoothing



    def fit(self, X, y):
        # No fitting needed
        return self

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        # Transform probabilities
        prob_pred = X - self.decision_threshold
        p0 = sigmoid(prob_pred) # Probability for class 1
        # Ensure probabilities sum to 1
        p1 = 1 - p0  # Probability for class 0
        probabilities = np.stack([p1, p0], axis=1)

        return probabilities

def kl_divergence(p, q, axis):
	# add epsilon for numeric stability
	p += 1e-10
	q += 1e-10
	return np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=axis)

# Add dropout to both g and h
class StochasticModel(nn.Module):
    def __init__(self, h, dropout_prob=0.5):
        super(StochasticModel, self).__init__()
        # self.g = nn.Sequential(
        #     g,
        #     nn.Dropout(p=dropout_prob)  # Add dropout after g
        # )
        self.h = nn.Sequential(
            nn.Dropout(p=dropout_prob),  # Add dropout before logits
            h
        )

    def forward(self, x):
        # x = self.g(x)
        self.h.train()
        x = self.h(x)
        return x

# Function to filter dataset by random classes
def filter_dataset_by_classes(dataset, num_classes, max_subset_size=None):
    """
    Filters a dataset to include only a subset of classes and limits the number of samples in the subset.

    Args:
        dataset (Dataset): The dataset to filter.
        num_classes (int): The number of classes to include in the subset.
        max_subset_size (int, optional): The maximum number of samples to include in the subset.
                                         If None, include all samples.

    Returns:
        Subset: A filtered subset of the dataset.
    """
    # Randomly select `num_classes` from the dataset's classes
    classes = random.sample(range(len(dataset.classes)), num_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Find indices of samples belonging to the selected classes
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]

    # If max_subset_size is specified, randomly select a subset of the indices
    if max_subset_size is not None and max_subset_size < len(indices):
        indices = random.sample(indices, max_subset_size)

    # Create the subset
    subset = Subset(dataset, indices)
    subset.classes = [dataset.classes[c] for c in classes]
    subset.class_to_idx = class_to_idx

    return subset

def predict_with_uncertainty_batched(f_model, data_loader, n_iter=10, device="cuda"):
    """
    Perform N stochastic forward passes over batches of data and return predictions.

    Args:
        f_model (nn.Module): The model with dropout layers.
        data_loader (DataLoader): DataLoader for the dataset.
        n_iter (int): Number of stochastic forward passes.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Predictions of shape (n_iter, num_data, num_classes).
    """
    # f_model.train()  # Ensure dropout is active during inference

    # Store predictions for all iterations
    all_preds = []
    for _ in range(n_iter):
        preds = []
        for inputs in data_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                batch_preds = torch.softmax(f_model(inputs), dim=1)

            preds.append(batch_preds)
        # Concatenate predictions for this iteration
        all_preds.append(torch.cat(preds, dim=0))

    # Stack predictions across iterations
    return torch.stack(all_preds, dim=0)


def assign_uncertainty_label(value, threshold):
    if value < threshold:
        return 0  # Low uncertainty
    else:  # (mean - std) <= value <= (mean + std):
        return 1
# Function to preprocess the images in batches to avoid OOM errors
def preprocess_images_in_batches(filelist,func,transform,to_pil, batch_size=64):
    all_images, all_labels = [], []
    for start_idx in range(0, len(filelist), batch_size):
        end_idx = min(start_idx + batch_size, len(filelist))
        batch_files = filelist[start_idx:end_idx]
        x, y = zip(*func(batch_files))
        x, y = np.array(x), np.array(y)
        images_preprocessed = torch.stack([transform(to_pil(img)) for img in x], 0)
        all_images.append(images_preprocessed)
        all_labels.append(torch.from_numpy(y))
    return torch.cat(all_images), torch.cat(all_labels)


class SimpleLinearNN(nn.Module):
    def __init__(self, input_features=768, output_features=9, dropout_prob=0.5):
        """
        Args:
            input_features (int): Number of input features.
            output_features (int): Number of output features (classes).
            dropout_prob (float): Probability of dropping out neurons.
        """
        super(SimpleLinearNN, self).__init__()
        self.linear = nn.Linear(in_features=input_features, out_features=output_features, bias=True)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.dropout(x)  # Apply dropout
        return self.linear(x)
def load_imagenette_single_class_images(target_class_name):
    device = 'cuda'

    # Load the model
    model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True)
    model = model.to(device)

    # Process the model's data configuration
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    to_pil = transforms.ToPILImage()

    # Generator function for images
    def gen_images(filelist):
        for f in filelist:
            folder_name = f.split('/')[-2]
            class_id = folder_name2class_id[folder_name]
            im = Image.open(f)
            if len(im.getbands()) == 3:
                yield np.array(im.resize((224, 224))), class_id

    # Load ImageNet class names
    with urllib.request.urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
        imagenet_class_names = np.array(f.read().decode('utf-8').split('\n'))

    # Load Imagenette dataset
    imagenette_folder = untar_data(URLs.IMAGENETTE_160)

    # Define Imagenette-specific folder and class mappings
    folders_names = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042',
                     'n03425413', 'n03445777', 'n03888257']
    imagenette_class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn',
                              'garbage truck', 'gas pump', 'golf ball', 'parachute', ]

    # Map folder names to ImageNet class IDs
    imagenette_class_ids = [np.where(imagenet_class_names == class_name)[0][0] for class_name in imagenette_class_names]
    folder_name2class_id = dict(zip(folders_names, imagenette_class_ids))

    # Check if the target class is valid
    if target_class_name not in imagenette_class_names:
        raise ValueError(f"Invalid class name '{target_class_name}'. Choose from: {imagenette_class_names}")

    # Get the folder name and class ID for the target class
    target_folder = folders_names[imagenette_class_names.index(target_class_name)]
    target_class_id = folder_name2class_id[target_folder]

    # Get file lists for the target class
    train_filelist = glob.glob(f'{imagenette_folder}/train/{target_folder}/*.JPEG')
    val_filelist = glob.glob(f'{imagenette_folder}/val/{target_folder}/*.JPEG')

    # Preprocess the images for the target class
    val_images, val_labels = preprocess_images_in_batches(val_filelist, gen_images, transform, to_pil)
    train_images, train_labels = preprocess_images_in_batches(train_filelist, gen_images, transform, to_pil)

    return train_images, train_labels, val_images, val_labels, model, imagenet_class_names[target_class_id], target_class_id


def load_imagewoof_single_class_images(target_class_name):
    device = 'cuda'

    # Load the model
    model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True)
    model = model.to(device)

    # Process the model's data configuration
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    to_pil = transforms.ToPILImage()

    # Generator function for images
    def gen_images(filelist):
        for f in filelist:
            folder_name = f.split('/')[-2]
            class_id = folder_name2class_id[folder_name]
            im = Image.open(f)
            if len(im.getbands()) == 3:
                yield np.array(im.resize((224, 224))), class_id

    # Load ImageNet class names
    with urllib.request.urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
        imagenet_class_names = np.array(f.read().decode('utf-8').split('\n'))

    # Load Imagenette dataset
    imagenette_folder = untar_data(URLs.IMAGEWOOF_160)

    # Define Imagenette-specific folder and class mappings

    folders_names = ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601',
                     'n02105641', 'n02111889', 'n02115641']
    imagenette_class_names = ['Shih-Tzu', 'Rhodesian ridgeback', 'beagle', 'English foxhound', 'Border terrier',
                              'Australian terrier', 'golden retriever', 'Old English sheepdog', 'Samoyed', 'dingo']

    # Map folder names to ImageNet class IDs
    imagenette_class_ids = [np.where(imagenet_class_names == class_name)[0][0] for class_name in imagenette_class_names]
    folder_name2class_id = dict(zip(folders_names, imagenette_class_ids))

    # Check if the target class is valid
    if target_class_name not in imagenette_class_names:
        raise ValueError(f"Invalid class name '{target_class_name}'. Choose from: {imagenette_class_names}")

    # Get the folder name and class ID for the target class
    target_folder = folders_names[imagenette_class_names.index(target_class_name)]
    target_class_id = folder_name2class_id[target_folder]

    # Get file lists for the target class
    train_filelist = glob.glob(f'{imagenette_folder}/train/{target_folder}/*.JPEG')
    val_filelist = glob.glob(f'{imagenette_folder}/val/{target_folder}/*.JPEG')

    # Preprocess the images for the target class
    val_images, val_labels = preprocess_images_in_batches(val_filelist, gen_images, transform, to_pil)
    train_images, train_labels = preprocess_images_in_batches(train_filelist, gen_images, transform, to_pil)

    return train_images, train_labels, val_images, val_labels, model, imagenet_class_names[target_class_id], target_class_id

def load_imagenette():
    device = 'cuda'

    # loading any timm model
    model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True)
    model = model.to(device)

    # processing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    to_pil = transforms.ToPILImage()

    # cut the model in two parts
    def gen_images(filelist):
        for f in filelist:
            folder_name = f.split('/')[-2]
            class_id = folder_name2class_id[folder_name]
            im = Image.open(f)
            if len(im.getbands()) == 3:
                yield np.array(im.resize((224, 224))), class_id

    with urllib.request.urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
        imagenet_class_names = np.array(f.read().decode('utf-8').split('\n'))

    imagenette_folder = untar_data(URLs.IMAGENETTE_160)

    folders_names = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042',
                     'n03425413', 'n03445777', 'n03888257']
    imagenette_class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn',
                              'garbage truck', 'gas pump', 'golf ball', 'parachute']

    imagenette_class_ids = [np.where(imagenet_class_names == class_name)[0][0] for class_name in imagenette_class_names]
    folder_name2class_id = dict(zip(folders_names, imagenette_class_ids))
    train_filelist = glob.glob(f'{imagenette_folder}/train/*/*.JPEG')
    val_filelist = glob.glob(f'{imagenette_folder}/val/*/*.JPEG')
    val_images, val_labels = preprocess_images_in_batches(val_filelist,gen_images, transform, to_pil)
    train_images, train_labels = preprocess_images_in_batches(train_filelist,gen_images, transform, to_pil)

    return train_images, train_labels, val_images, val_labels, model

# Define a dataset class for embeddings
class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class LabelledDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, labels):
            """
            Initialize the dataset with original data and new labels.

            Args:
                original_dataset (Dataset): The original dataset.
                labels (torch.Tensor): The new labels for the dataset.
            """
            self.original_dataset = original_dataset
            self.labels = labels

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            data, _ = self.original_dataset[idx]  # Ignore the original labels
            label = self.labels[idx]
            return data, label

# Function to sample datasets
def sample_datasets(iid_dataset, ood_dataset, iid_count, ood_percentage):
    iid_indices = np.random.choice(len(iid_dataset), iid_count)
    iid_subset = Subset(iid_dataset, torch.from_numpy(iid_indices).long())

    # Sample OOD dataset
    # ood_count = int(len(ood_dataset) * ood_percentage)
    # ood_indices = np.random.choice(len(ood_dataset), ood_count)
    # ood_subset = Subset(ood_dataset, ood_indices)

    # Merge datasets and create labels
    combined_dataset = ConcatDataset([iid_subset, ood_dataset])
    iid_labels = torch.zeros(len(iid_subset))
    ood_labels = torch.ones(len(ood_dataset))
    labels = torch.cat([iid_labels, ood_labels]).long()

    return combined_dataset, labels
def load_imagewoof():
    device = 'cuda'

    # loading any timm model
    model = timm.create_model('nf_resnet50.ra2_in1k', pretrained=True)
    model = model.to(device)

    # processing
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    to_pil = transforms.ToPILImage()

    # cut the model in two parts
    g = nn.Sequential(*(list(model.children())[:4]))
    def gen_images(filelist):
        for f in filelist:
            folder_name = f.split('/')[-2]
            class_id = folder_name2class_id[folder_name]
            im = Image.open(f)
            if len(im.getbands()) == 3:
                yield np.array(im.resize((224, 224))), class_id

    with urllib.request.urlopen('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt') as f:
        imagenet_class_names = np.array(f.read().decode('utf-8').split('\n'))

        # Load imagewoof dataset
    imagewoof_folder = untar_data(URLs.IMAGEWOOF_160)
    folders_names = ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601',
                     'n02105641', 'n02111889', 'n02115641']
    imagenette_class_names = ['Shih-Tzu', 'Rhodesian ridgeback', 'beagle', 'English foxhound', 'Border terrier',
                              'Australian terrier', 'golden retriever', 'Old English sheepdog', 'Samoyed', 'dingo']
    imagenette_class_ids = [np.where(imagenet_class_names == class_name)[0][0] for class_name in imagenette_class_names]

    folders_names = ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601',
                         'n02105641', 'n02111889', 'n02115641']

    imagewoof_class_ids = list(range(0, 10))
    folder_name2class_id = dict(zip(folders_names, imagewoof_class_ids))

        # Get file lists for train, val, and test sets
    train_filelist = glob.glob(f'{imagewoof_folder}/train/*/*.JPEG')
    val_filelist = glob.glob(f'{imagewoof_folder}/val/*/*.JPEG')
    val_images, val_labels = preprocess_images_in_batches(val_filelist,gen_images, transform, to_pil)
    train_images, train_labels = preprocess_images_in_batches(train_filelist, gen_images, transform, to_pil)

    return train_images, train_labels, val_images, val_labels,model


def nmf_transform(inputs, drift_basis, patches=False, labels=None, activations=None, n_patches=16):
    """
    Transforms the input images into an (N, 320) representation where N is the number of images.

    Parameters:
    - inputs: Input images or data to be transformed.
    - patches: Whether to use patches (if needed for some other functionality).
    - labels: Optional labels for the inputs.
    - activations: Optional pre-computed activations. If None, activations are computed.
    - drift_basis: Predefined basis for NMF.
    - n_patches: Number of patches per image (default is 16).

    Returns:
    - transformed_data: Transformed dataset with shape (N, 320).
    """

    # Step 1: Extract latent activations using drift_craft
    A = np.mean(inputs, axis=(
    1, 2))  # drift_craft._latent_predict(inputs)  # Assuming A.shape = (N, H, W, D) where D is the activation dimension

    # Step 3: Apply Non-negative Matrix Factorization (NMF) to reduce dimensionality
    embedding, basis, n_iter = non_negative_factorization(A,
                                                          n_components=len(drift_basis),
                                                          init='custom',
                                                          update_H=False, solver='mu', H=drift_basis)
    # print(embedding.shape)

    return embedding


def predict_with_uncertainty_batched_sig(f_model, data_loader, n_iter=10, device="cpu"):
    """
    Perform N stochastic forward passes over batches of data and return predictions.

    Args:
        f_model (nn.Module): The model with dropout layers.
        data_loader (DataLoader): DataLoader for the dataset.
        n_iter (int): Number of stochastic forward passes.
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Predictions of shape (n_iter, num_data, num_classes).
    """
    # f_model.train()  # Ensure dropout is active during inference

    # Store predictions for all iterations
    all_preds = []
    for _ in range(n_iter):
        preds = []
        for inputs in data_loader:
            with torch.no_grad():
                batch_preds = torch.softmax(f_model(inputs), dim=1)

            preds.append(batch_preds)
        # Concatenate predictions for this iteration
        all_preds.append(torch.cat(preds, dim=0))

    # Stack predictions across iterations
    return torch.stack(all_preds, dim=0)



def accuracy_rejection(predictions, labels, gt_idd_ood, uncertainty, step_length=90, plot=True, plot_random=False,
                       directory="./pic/", plot_name="acc_rej", log=False):
    accuracy_list = []
    r_accuracy_list = []
    odd = []
    # for uncertainty in uncertainty_list:
    correctness_map = []
    for x, y in zip(predictions, labels):
        if x == y:
            correctness_map.append(1)
        else:
            correctness_map.append(0)

    # uncertainty, correctness_map = zip(*sorted(zip(uncertainty,correctness_map),reverse=False))

    correctness_map = np.array(correctness_map)
    sorted_index = np.argsort(uncertainty, kind='stable')
    uncertainty = uncertainty[sorted_index]
    correctness_map = correctness_map[sorted_index]
    odd_map = gt_idd_ood[sorted_index]

    odd_map = list(odd_map)
    correctness_map = list(correctness_map)
    uncertainty = list(uncertainty)
    data_len = len(correctness_map)
    accuracy = []
    steps = list(range(step_length))
    for x in steps:
        rejection_index = int(data_len * (len(steps) - x) / len(steps))
        x_correct = correctness_map[:rejection_index].copy()
        x_unc = uncertainty[:rejection_index].copy()
        x_odd = odd_map[:rejection_index].copy()
        if log:
            print(f"----------------------------------------------- rejection_index {rejection_index}")
            for c, u, o in zip(x_correct, x_unc, x_odd):
                print(f"correctness_map {c} uncertainty {u} odd {o}")
        # print(f"rejection_index = {rejection_index}\nx_correct {x_correct} \nunc {x_unc}")
        if rejection_index == 0:
            accuracy.append(np.nan)  # random.random()
        else:
            accuracy.append(np.sum(x_correct) / rejection_index)
            odd.append(np.sum(x_odd) / rejection_index)
    accuracy_list.append(accuracy)

    # random test plot
    r_accuracy = []

    for x in steps:
        random.shuffle(correctness_map)
        r_rejection_index = int(data_len * (len(steps) - x) / len(steps))
        r_x_correct = correctness_map[:r_rejection_index].copy()
        if r_rejection_index == 0:
            r_accuracy.append(np.nan)
        else:
            r_accuracy.append(np.sum(r_x_correct) / r_rejection_index)

    r_accuracy_list.append(r_accuracy)

    accuracy_list = np.array(accuracy_list)
    r_accuracy_list = np.array(r_accuracy_list)

    avg_accuracy = np.nanmean(accuracy_list, axis=0)
    avg_r_accuracy = np.nanmean(r_accuracy_list, axis=0)
    # max_acc = np.amax(accuracy_list, axis=0)
    # min_acc = np.amin(accuracy_list, axis=0)
    std_error = np.std(accuracy_list, axis=0) / math.sqrt(len(uncertainty))

    if plot:
        plt.plot(steps, avg_accuracy, label="")  # , marker='o', linestyle='--' , color='r'
        if plot_random:
            plt.plot(steps, avg_r_accuracy, label="Random")
        plt.fill_between(steps, avg_accuracy + std_error, avg_accuracy - std_error, alpha=0.5)

        plt.xlabel('Rejection %')
        plt.ylabel('Accuracy %')
        plt.title(plot_name)  # 'Accuracy-Rejection curve'
        plt.legend()

    return avg_accuracy, avg_accuracy - std_error, avg_accuracy + std_error, avg_r_accuracy, odd, steps


class ClassifierHead(nn.Module):
    def __init__(self, in_features=2048, out_features=10):
        super(ClassifierHead, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features, out_features)
        self.flatten = nn.Identity()

    def forward(self, x):
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

def process_in_batches(data, batch_size, func, g):
    """
    Process data in batches using the full_wrapper function.

    Args:
        data (np.ndarray): Input data to process.
        batch_size (int): Size of each batch.

    Returns:
        np.ndarray: Predictions for the entire dataset.
    """
    # Initialize an empty list to store predictions
    all_preds = []

    # Iterate over the data in batches
    for start_idx in range(0, len(data), batch_size):
        # Get the end index for the current batch
        end_idx = min(start_idx + batch_size, len(data))
        # Get the current batch
        batch = data[start_idx:end_idx]
        # Use full_wrapper to process the batch and get predictions
        batch_preds = func(batch,g)
        # Append the predictions to the list
        all_preds.append(batch_preds)

    # Concatenate all batch predictions into a single numpy array
    all_preds = np.concatenate(all_preds, axis=0)
    return torch.from_numpy(all_preds)

class SimpleNN(nn.Module):
    def __init__(self,):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class SimpleUncNN(nn.Module):
    def __init__(self,out_features=2):
        super(SimpleUncNN, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Train the neural network
def train_nn(train_loader, model, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Evaluate the model
from sklearn.metrics import f1_score, matthews_corrcoef

def evaluate_nn(test_loader, model):
    """
    Evaluate the model on the test data and compute accuracy and F1 score.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model (nn.Module): The trained model.

    Returns:
        dict: A dictionary containing accuracy and F1 score.
    """
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            outputs = model(features)
            _, predicted = torch.max(torch.softmax(outputs, dim=1), 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store labels and predictions for F1 score
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = correct / total

    # Calculate F1 score (binary)
    f1 = f1_score(all_labels, all_predictions, average="binary")
    mcc = matthews_corrcoef(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    return accuracy#, # , mcc


class UncCNN(nn.Module):
    def __init__(self, out_features=2):
        super(UncCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
