import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import re
import scipy

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

from concept_helpers.cockatiel_sub import SubCockatiel

from tqdm import tqdm
from typing import  Dict
from cockatiel.cockatiel.sampling import ScipySobolSequence
from cockatiel.cockatiel.sobol import JansenEstimator
from experiment_helpers.experiment_helper_functions import *

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

# Updated function for excerpt extraction
def excerpt_fct(raw_dataset):
    excerpt_dataset = []
    
    # Regex to capture sentence boundaries (handles '.', '!', '?', etc.)
    sentence_pattern = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)

    for review in raw_dataset[:100000]:
        review = str(review)

        # Split into sentences based on the pattern
        sentences = sentence_pattern.findall(review)
        
        for sentence in sentences:
            # Clean up leading/trailing spaces
            sentence = sentence.strip()

            # Ensure the sentence starts with an uppercase letter
            if sentence and sentence[0].isupper():
                excerpt_dataset.append(sentence)

    return excerpt_dataset

def compute_predictions_sigmoid(inputs, stochastic_model, device='cuda'):

    inputs = inputs.to(device)
    data_loader = torch.utils.data.DataLoader(inputs, batch_size=64)
    
    predictions = predict_with_uncertainty_batched(stochastic_model, data_loader, n_iter=100, device=device)

    predictions =  predictions.cpu().numpy()
    
    a,prob_mat = uncertainty_matrices(predictions)
    t,e,a = entropy_uncertainty(prob_mat)
    sig_threshold, _, t_norm = get_threshold(t)
    loc = UncertaintyWrapperWithSigmoid(sig_threshold)

    unc_pred_probs = loc.predict_proba(t_norm)
    unc_preds = np.argmax(unc_pred_probs, axis=1)
    return unc_preds, unc_pred_probs

class Sensitivity:
    """
    Dataclass handling data produced during the Sobol indices computation.
    This is an internal data class used by Craft to store computation data.

    Parameters
    ----------
    importances
        The Sobol total index (importance score) for each concept.
    most_important_concepts
        The number of concepts to display. If None is provided, then all the concepts
        will be displayed unordered, otherwise only nb_most_important_concepts will be
        displayed, ordered by importance.
    cmaps
        The list of colors associated with each concept.
        Can be either:
            - A list of (r, g, b) colors to use as a base for the colormap.
            - A colormap name compatible with `plt.get_cmap(cmap)`.
    """

    def __init__(self, importances: np.ndarray,
                       most_important_concepts: np.ndarray,
                       # cmaps: Optional[Union[list, str]]=None
                ):
        self.importances = importances
        self.most_important_concepts = most_important_concepts
       
        
def estimate_importance(cockatiel_instance, stochastic_model, inputs, class_of_interest, W:np.ndarray):
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
    # embeddings = bert.embed(inputs)
    # y_preds, _ = compute_predictions_sigmoid(inputs)
    # print(activations.shape)

    global_importance = []
    sensitivities = {}
    # for class_of_interest in np.unique(y_preds):
    # filtered_indices = np.where(y_preds == class_of_interest)
    class_inputs = inputs

    importances = _drift_sent_importance(cockatiel_instance, 
                                         stochastic_model, 
                                         class_inputs, 
                                         class_of_interest, 
                                         W=W,
                                         sens_dict=sensitivities,
                                         compute_class_importance=True, 
                                        )
    global_importance.append(importances)
        

    return global_importance, sensitivities

def _drift_sent_importance(cockatiel_instance, stochastic_model, dataset, class_id: int, W:np.ndarray,sens_dict= None, compute_class_importance=False, device='cuda'):
    """
    

    Parameters
    ----------
    cropped_dataset
        The activations of the dataset containing the excerpts used to discover the concepts.
   
    class_id
        An integer for the class we wish to explain.
    W
        The (already learned) concept base.

    Returns
    -------
    global_importance
        An array with the Sobol indices
    """
    masks = ScipySobolSequence()(len(W), nb_design=cockatiel_instance.sobol_nb_design)
    # print(masks.shape)
    estimator = JansenEstimator()
    
    activations = activation_transform(dataset,W)#bert_nmf_transform(dataset,W)
    

    # if not isinstance(W, torch.Tensor):
    #     W = torch.Tensor(W).float().to(cockatiel_explainer_autos.device)

    importances = []
    #i = 0
    for act in tqdm(activations):
        #print(i)
        #i += 1
        act = torch.Tensor(act).float().to(cockatiel_instance.device)

        # y_pred = None
        # for batch_id in range(ceil(len(cropped_dataset) / cockatiel_instance.batch_size)):
        #     batch_start = batch_id * cockatiel_instance.batch_size
        #     batch_end = batch_start + cockatiel_instance.batch_size
        #     batch_masks = torch.Tensor(masks[batch_start:batch_end]).float().to(cockatiel_instance.device)

        y_pred = concept_perturbation(stochastic_model, act, 
                                        torch.from_numpy(masks).to(cockatiel_instance.device), class_id, W)
        #print("ypred: ", y_pred)
        # y_pred = y_batch if y_pred is None else torch.cat([y_pred, y_batch], 0)

        # if cockatiel_instance.device == 'cuda' or cockatiel_instance.device == torch.device('cuda'):
        y_pred = y_pred.cpu()
        stis = estimator(masks, y_pred.numpy(), cockatiel_instance.sobol_nb_design)
        #print("stis: ", stis)
        importances.append(stis)

    global_importance = np.mean(importances, 0)
    
    if compute_class_importance:
            most_important_concepts = np.argsort(global_importance)[::-1]
            sens_dict[class_id] = Sensitivity(global_importance, most_important_concepts)

    return global_importance

def concept_perturbation(stochastic_model, activation, masks, class_id, W):
    """
    Apply perturbation on the concept before reconstruction and get the perturbated outputs.
    For NMF we recall that A = U @ W

    Parameters
    ----------
    model
      Model that map the concept layer to the output (h_l->k in the paper)
    activation
      Specific activation to apply perturbation on.
    masks
      Arrays of masks, each of them being a concept perturbation.
    class_id
      Id the class to test.
    W
      Concept bank extracted using NMF.

    Returns
    -------
    y
      Outputs of the perturbated points.
    """
    # model = model.model

    perturbation = masks #@ W
    # print(masks.shape)

    if len(activation.shape) == 3:
        perturbation = perturbation[:, None, None, :]

    # print(perturbation.shape)
    activation = activation[None, :]
    # print(activation.shape)
    perturbated_activations = activation * perturbation
    # print(perturbated_activations.shape)
    perturbated_activations = perturbated_activations.detach().cpu().numpy() @ W
    _, probs = compute_predictions_sigmoid(torch.from_numpy(perturbated_activations), stochastic_model)
    y = probs[:, class_id]
    # print(y)

    return torch.from_numpy(y)
    
def activation_transform(inputs, drift_basis, patches=False, labels=None, activations=None, n_patches=16):
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
    # image_size = (inputs.shape[1], inputs.shape[2])

    # patches, patch_act, train_labels = h_craftdv._extract_patches(np.reshape(inputs, (1,3,256,256)), labels)
    # Step 1: Extract latent activations using drift_craft
    A = inputs.cpu().numpy()
    
    # Step 3: Apply Non-negative Matrix Factorization (NMF) to reduce dimensionality
    embedding, basis, n_iter = non_negative_factorization(A,
                                                          n_components=len(drift_basis),
                                                          init='custom',
                                                          update_H=False, solver='mu', H=drift_basis)
   

    return embedding

# Create stochastic model for MC dropout; must be torch; h refers to classification head 
class StochasticModel(nn.Module):
    def __init__(self, h, dropout_prob=0.5):
        super(StochasticModel, self).__init__()
     
        self.h = nn.Sequential(
            # g,
            nn.Dropout(p=dropout_prob),  # Add dropout before logits
            h
        )

    def forward(self, x):
        self.h.train()
        x = self.h(x)
        return x

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
                batch_preds = f_model(inputs)
                # print(batch_preds)
            if len(batch_preds.size()) == 1:
                preds.append(batch_preds.reshape((1,batch_preds.shape[0])))
            else:
                preds.append(batch_preds)
        # Concatenate predictions for this iteration
        all_preds.append(torch.cat(preds, dim=0))

    # Stack predictions across iterations
    return torch.stack(all_preds, dim=0)
