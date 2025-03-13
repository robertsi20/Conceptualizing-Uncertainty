import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cockatiel.cockatiel import COCKATIEL

import torch
from sklearn.decomposition import NMF
from math import ceil

from cockatiel.cockatiel.sampling import ScipySobolSequence, concept_perturbation
from cockatiel.cockatiel.sobol import JansenEstimator
from cockatiel.cockatiel.utils import tokenize

from typing import Callable, List

import numpy as np


class SubCockatiel(COCKATIEL):
    """
        A class implementing COCKATIEL, the concept based explainability method for NLP introduced
        in https://arxiv.org/abs/2305.06754

        Parameters
        ----------
        model
            The Torch hugging-face model that we wish to explain. It MUST have a non-negative layer on
            which to extract the concepts.
        tokenizer
            A callable object to transform strings into inputs for the model.
        components
            An integer for the amount of concepts we wish to discover in the activation space.
        batch_size
            The batch size for all the operations that use the model
        device
            The type of device on which to place the torch tensors
        """
    sobol_nb_design = 32

    def __init__(
            self,
            model,
            tokenizer: Callable,
            components: int = 25,
            batch_size: int = 256,
            device: str = 'cuda',
            nmf_max_iter: int = 1000
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.components = components
        self.batch_size = batch_size
        self.device = device
        self.max_iter = nmf_max_iter
        self.W = None
        super().__init__(model=model, tokenizer=tokenizer, components=components)

    def extract_concepts(
            self,
            cropped_dataset: List[str],
            cropped_embeddings,
            dataset: List[str],
            embeddings,
            class_id: int,
            limit_sobol: int = 1000,
            activation_layer = None
    ):
        """
        Extracts the concepts following the object's parameters.

        Parameters
        ----------
        cropped_dataset
            The dataset containing the excerpts used to discover the concepts.
        dataset
            A sample of the dataset (with whole inputs) on which to compute the Sobol importance.
        class_id
            An integer for the class we wish to explain.
        limit_sobol
            The maximum amount of masks to use for estimating Sobol indices.

        Returns
        -------
        excerpts
            The excerpts used to learn the concepts.
        u_excerpts
            The coefficients in the learned concept base for the excerpts.
        factorization
            The object to transform activations using the concept base.
        global_importance
            An array with the global importance of each concept (Sobol indices).
        """
        excerpts = []
        excerpts_activations = None

        with torch.no_grad():
            for batch_id in range(ceil(len(cropped_dataset) / self.batch_size)):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                batch_sentences = cropped_dataset[batch_start:batch_end]
                # batch_tokenized = tokenize(batch_sentences, self.tokenizer, self.device)

                batch_activations = cropped_embeddings[batch_start:batch_end,:] # using embeddings directly now  #self.model.embed(batch_sentences)
                batch_activations = torch.from_numpy(batch_activations)
                if activation_layer is not None:
                    batch_activations = activation_layer(batch_activations)
                excerpts_activations = batch_activations if excerpts_activations is None \
                    else torch.cat([excerpts_activations, batch_activations], 0)
                excerpts = batch_sentences if excerpts is None \
                   else excerpts + batch_sentences

        points_activations = None
        with torch.no_grad():
            for batch_id in range(ceil(len(dataset) / self.batch_size)):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size
                # tokenized_batch = tokenize(dataset[batch_start:batch_end], self.tokenizer, self.device)
                activations = embeddings[batch_start:batch_end] # using embeddings directly now #self.model.embed(dataset[batch_start:batch_end])
                activations = torch.from_numpy(activations)
                if activation_layer is not None:
                    activations = activation_layer(activations)
                points_activations = activations if points_activations is None else torch.cat(
                    [points_activations, activations])

        # applying GAP(.) on the activation and ensure positivity if needed
        excerpts_activations = self._preprocess(excerpts_activations)
        points_activations = self._preprocess(points_activations)

        # using the activations, we will now use the matrix factorization to
        # find the concept bank (W) and the concept representation (U) of the
        # segments and the points
        factorization = NMF(n_components=self.components, max_iter=self.max_iter)

        u_excerpts = factorization.fit_transform(excerpts_activations)
        self.W = torch.Tensor(factorization.components_).float().to(self.device)

        # we don't need segments activations anymore, the concept bank is trained
        del excerpts_activations

        # using the concept bank and the points, we will now evaluate the importance of
        # each concept for each points to get a global importance score for each
        # concept in the concept bank
        global_importance = self._sobol_importance(cropped_dataset, points_activations[:limit_sobol], class_id, self.W)

        return excerpts, u_excerpts, factorization, global_importance, points_activations[:limit_sobol]

    def _sobol_importance(self, cropped_dataset, activations: torch.Tensor, class_id: int, W: torch.Tensor):
        """
        Computes the Sobol indices using the dataset containing the excerpts and the activations from the
        dataset (whole inputs) for the target class, and for a fixed (already learned) concept base W.

        Parameters
        ----------
        cropped_dataset
            The activations of the dataset containing the excerpts used to discover the concepts.
        activations
            The activations for inputs from the original dataset.
        class_id
            An integer for the class we wish to explain.
        W
            The (already learned) concept base.

        Returns
        -------
        global_importance
            An array with the Sobol indices
        """
        masks = ScipySobolSequence()(self.components, nb_design=self.sobol_nb_design)
        estimator = JansenEstimator()

        if not isinstance(W, torch.Tensor):
            W = torch.Tensor(W).float().to(self.device)

        importances = []
        for act in activations:
            act = torch.Tensor(act).float().to(self.device)

            y_pred = None
            for batch_id in range(ceil(len(cropped_dataset) / self.batch_size)):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size
                batch_masks = torch.Tensor(masks[batch_start:batch_end]).float().to(self.device)
                # print(batch_masks.shape)

                y_batch = self.concept_perturbation(self.model, act, batch_masks, class_id, W)
                y_pred = y_batch if y_pred is None else torch.cat([y_pred, y_batch], 0)

            if self.device == 'cuda' or self.device == torch.device('cuda'):
                y_pred = y_pred.cpu()
            stis = estimator(masks, y_pred.numpy(), self.sobol_nb_design)
            importances.append(stis)

        global_importance = np.mean(importances, 0)

        return global_importance


    def concept_perturbation(self, model, activation, masks, class_id, W):
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
        model = model.model

        perturbation = masks @ W

        if len(activation.shape) == 3:
            perturbation = perturbation[:, None, None, :]

        activation = activation[None, :]
        perturbated_activations = activation + perturbation * activation
        y = model.classifier(perturbated_activations)[:, class_id]

        return y