from xplique.concepts import CraftTf
from typing import Callable, Optional, Tuple
from math import ceil
from matplotlib import gridspec, pyplot as plt
from xplique.attributions.global_sensitivity_analysis import HaltonSequenceRS, JansenEstimator
from xplique.concepts.craft import BaseCraft, DisplayImportancesOrder, Factorization, Sensitivity

import tensorflow as tf
import numpy as np
from sklearn.decomposition import non_negative_factorization

from xplique.plots.image import _clip_percentile
import cv2


class CombinedCraftsTf(CraftTf):
    """
        Class implementing the CRAFT Concept Extraction Mechanism on Tensorflow.

        Parameters
        ----------
        input_to_latent_model
            The first part of the model taking an input and returning
            positive activations, g(.) in the original paper.
            Must be a Tensorflow model (tf.keras.engine.base_layer.Layer) accepting
            data of shape (n_samples, height, width, channels).
        latent_to_logit_model
            The second part of the model taking activation and returning
            logits, h(.) in the original paper.
            Must be a Tensorflow model (tf.keras.engine.base_layer.Layer).
        number_of_concepts
            The number of concepts to extract. Default is 20.
        batch_size
            The batch size to use during training and prediction. Default is 64.
        patch_size
            The size of the patches to extract from the input data. Default is 64.
        """

    def __init__(self, input_to_latent_model: Callable,
                 latent_to_logit_model: Callable,
                 number_of_concepts: int = 20,
                 inputs=np.ndarray,
                 labels=np.ndarray,
                 basis = np.ndarray,
                 batch_size: int = 64,
                 patch_size: int = 64):
        super().__init__(input_to_latent_model,
                         latent_to_logit_model,
                         number_of_concepts,
                         batch_size)
        self.patch_size = patch_size
        self.basis = basis
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.inputs = inputs
        self.labels = labels
        self.basis = basis
        self.classes_of_interest = np.unique(labels)
        self.num_classes = len(np.unique(labels))
        self.components_per_class = number_of_concepts
        self.number_of_concepts = number_of_concepts
        self.n_components = len(basis)
        self.craft_instances = {}
        self.sensitivities = {}
        self.transform_num_data_dict = {}

        # Check model type
        keras_base_layer = tf.keras.Model

        is_tf_model = issubclass(type(input_to_latent_model), keras_base_layer) & \
                      issubclass(type(latent_to_logit_model), keras_base_layer)
        if not is_tf_model:
            raise TypeError('input_to_latent_model and latent_to_logit_model are not ' \
                            'Tensorflow models')

    def transform_all(self):

        crops, activations = self._extract_patches(self.inputs)

        embedding, basis, n_iter = non_negative_factorization(np.array(activations), n_components=self.n_components,
                                                              init='custom',
                                                              update_H=False, solver='mu', H=self.basis)

        self.factorization = Factorization(None, None, crops,
                                           None, embedding, self.basis)

    def transform(self, inputs, labels = None, activations = None):
        A = self._latent_predict(inputs)
        original_shape = A.shape[:-1]
        re_activations = np.reshape(A, (-1, A.shape[-1]))
        embedding, basis, n_iter = non_negative_factorization(np.array(re_activations), n_components=self.n_components,
                                                              init='custom',
                                                              update_H=False, solver='mu', H=self.basis)
        embedding = np.reshape(embedding, (*original_shape, embedding.shape[-1]))


        return embedding

    def plot_image_concepts(self,
                            img: np.ndarray,
                            img_local_importance: np.ndarray,
                            yt: int,
                            yp: int,
                            display_importance_order: DisplayImportancesOrder = \
                                    DisplayImportancesOrder.GLOBAL,
                            nb_most_important_concepts: int = 5,
                            filter_percentile: int = 90,
                            clip_percentile: Optional[float] = 10,
                            alpha: float = 0.65,
                            filepath: Optional[str] = None,
                            **plot_kwargs):
        """
        All in one method displaying several plots for the image `id` given in argument:
        - the concepts attribution map for this image
        - the best crops for each concept (displayed around the heatmap)
        - the importance of each concept

        Parameters
        ----------
        img
            The image to display.
        display_importance_order
            Selects the order in which the concepts will be displayed, either following the
            global importance on the whole dataset (same order for all images) or the local
            importance of the concepts for a single image sample (local importance).
            Default to GLOBAL.
        nb_most_important_concepts
            The number of concepts to display. Default is 5.
        filter_percentile
            Percentile used to filter the concept heatmap
            (only show concept if excess N-th percentile). Defaults to 90.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            Default to 10.
        alpha
            The alpha channel value for the heatmaps. Defaults to 0.65.
        filepath
            Path the file will be saved at. If None, the function will call plt.show().
        plot_kwargs
            Additional parameters passed to `plt.imshow()`.
        """
        fig = plt.figure(figsize=(20, 8))

        if display_importance_order == DisplayImportancesOrder.LOCAL:
            # compute the importances for the sample input in argument
            importances = self.estimate_importance(inputs=img)
            most_important_concepts = np.argsort(importances)[::-1][:nb_most_important_concepts]
        else:
            # use the global importances computed on the whole dataset
            importances = self.sensitivities[yp].importances
            most_important_concepts = \
                self.sensitivities[yp].most_important_concepts[:nb_most_important_concepts]

        # create the main gridspec which is split in the left and right parts storing
        # the crops, and the central part to display the heatmap
        nb_rows = ceil(len(most_important_concepts) / 2.0)
        nb_cols = 4
        gs_main = fig.add_gridspec(nb_rows, nb_cols, wspace=.2, hspace=0.4, width_ratios=[0.2, 0.4, 0.2, 0.4])

        # Add ghost axes and titles on gs1 and gs2
        s = r'$\mathbf{True\ Label:}\ $' + r'$\mathbf{' + str(yt) + '}$' + '\n' + \
            r'$\mathit{Predicted\ Label:}\ $' + r'$\mathit{' + str(yp) + '}$'

        fig.text(.55, .2, s, fontsize=14, fontfamily='serif', color='darkblue', ha='center', va='center')
        # s = 'True Label:' + str(yt) + '\nPredicted Label:' + str(yp)
        # fig.text(.55,.2,s)
        #
        # ax.set_title('True Label:' + str(yt) + '\nPredicted Label:' + str(yp))
        # Central image
        #
        fig.add_subplot(gs_main[:2, 1])
        self.plot_concept_attribution_map(image=img,
                                          most_important_concepts=np.argsort(img_local_importance)[::-1][:nb_most_important_concepts],
                                          class_concepts=yp,
                                          nb_most_important_concepts=nb_most_important_concepts,
                                          filter_percentile=filter_percentile,
                                          clip_percentile=clip_percentile,
                                          alpha=alpha,
                                          **plot_kwargs)

        fig.add_subplot(gs_main[2, 1])
        self.plot_concepts_importances(class_concepts=yp,
                                       importances=img_local_importance,
                                       display_importance_order=DisplayImportancesOrder.LOCAL,
                                       nb_most_important_concepts=nb_most_important_concepts,
                                       verbose=False)

        # Concepts: creation of the axes on left and right of the image for the concepts
        #
        gs_concepts_axes = [gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[i, 0])
                            for i in range(nb_rows)]
        gs_right = [gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_main[i, 2])
                    for i in range(nb_rows)]
        gs_concepts_axes.extend(gs_right)

        # display the best crops for each concept, in the order of the most important concept
        nb_crops = 6

        # compute the right color to use for the crops
        # global_color_index_order = np.argsort(self.sensitivities[yp].importances)[::-1]
        local = np.argsort(img_local_importance)[::-1]
        local_color_index_order = [np.where(local == local_c)[0][0]
                                   for local_c in local[:nb_most_important_concepts]]
        local_cmap = np.array(self.sensitivities[yp].cmaps)[local_color_index_order]

        for i, c_id in enumerate(local[:nb_most_important_concepts]):
            cmap = local_cmap[i]

            # use a ghost invisible subplot only to have a border around the crops
            ghost_axe = fig.add_subplot(gs_concepts_axes[i][:, :])
            ghost_axe.set_title(f"{c_id}", color=cmap(1.0))
            ghost_axe.axis('off')

            inset_axes = ghost_axe.inset_axes([-0.04, -0.04, 1.08, 1.08])  # outer border
            inset_axes.set_xticks([])
            inset_axes.set_yticks([])
            for spine in inset_axes.spines.values():  # border color
                spine.set_edgecolor(color=cmap(1.0))
                spine.set_linewidth(3)

            # draw each crop for this concept
            gs_current = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=
            gs_concepts_axes[i][:, :])

            best_crops_ids = np.argsort(self.factorization.crops_u[:, c_id])[::-1][:nb_crops]
            best_crops = np.array(self.factorization.crops)[best_crops_ids]
            for i in range(nb_crops):
                axe = plt.Subplot(fig, gs_current[i // 3, i % 3])
                fig.add_subplot(axe)
                BaseCraft._show(best_crops[i])

        # Right plot: importances
        importance_axe = gridspec.GridSpecFromSubplotSpec(3, 2, width_ratios=[0.1, 0.9],
                                                          height_ratios=[0.15, 0.6, 0.15],
                                                          subplot_spec=gs_main[:, 3])
        fig.add_subplot(importance_axe[1, 1])
        self.plot_concepts_importances(class_concepts=yp,
                                       importances=importances,
                                       display_importance_order=display_importance_order,
                                       nb_most_important_concepts=nb_most_important_concepts,
                                       verbose=False)

        if filepath is not None:
            plt.savefig(filepath)
        else:
            plt.show()

    def plot_concept_attribution_map(self,
                                     image: np.ndarray,
                                     most_important_concepts: np.ndarray,
                                     class_concepts: int,
                                     nb_most_important_concepts: int = 5,
                                     filter_percentile: int = 90,
                                     clip_percentile: Optional[float] = 10,
                                     alpha: float = 0.65,
                                     **plot_kwargs):
        """
        Display the concepts attribution map for a single image given in argument.

        Parameters
        ----------
        image
            The image to display.
        most_important_concepts
            The concepts ids to display.
        nb_most_important_concepts
            The number of concepts to display. Default is 5.
        filter_percentile
            Percentile used to filter the concept heatmap.
            (only show concept if excess N-th percentile). Defaults to 90.
        clip_percentile
            Percentile value to use if clipping is needed when drawing the concept,
            e.g a value of 1 will perform a clipping between percentile 1 and 99.
            This parameter allows to avoid outliers in case of too extreme values.
            It is applied after the filter_percentile operation.
            Default to 10.
        alpha
            The alpha channel value for the heatmaps. Defaults to 0.65.
        plot_kwargs
            Additional parameters passed to `plt.imshow()`.
        """
        # pylint: disable=E1101
        most_important_concepts = most_important_concepts[:nb_most_important_concepts]

        # Find the colors corresponding to the importances
        global_color_index_order = np.argsort(self.sensitivities[class_concepts].importances)[::-1]
        local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                   for local_c in most_important_concepts]
        # local_cmap = np.array(self.sensitivities[class_concepts].cmaps)[local_color_index_order]

        if class_concepts == 0:
            color_tup = [plt.get_cmap('tab20c').colors[0], plt.get_cmap('tab20c').colors[1],
                         plt.get_cmap('tab20c').colors[2],
                         plt.get_cmap('tab20c').colors[3], plt.get_cmap('tab20').colors[19]]
            for i in range(25):
                color_tup.append((plt.get_cmap('tab20c').colors + plt.get_cmap('Set3').colors)[i])

            drift_cmap = np.array([Sensitivity._get_alpha_cmap(cmap) for cmap in color_tup])
            local_cmap = drift_cmap[local_color_index_order]

            # local_cmap = np.array([colors(1.0)
            #                    for colors in drift_cmap])[local_color_index_order]
        if class_concepts == 1:
            color_tup = [plt.get_cmap('tab20b').colors[12], plt.get_cmap('tab20b').colors[13],
                         plt.get_cmap('tab20b').colors[14],
                         plt.get_cmap('tab20b').colors[15], plt.get_cmap('tab20').colors[6]]
            for i in range(25):
                color_tup.append((plt.get_cmap('tab20c').colors + plt.get_cmap('Set3').colors)[i])

            drift_cmap =  np.array([Sensitivity._get_alpha_cmap(cmap) for cmap in color_tup])
            local_cmap = drift_cmap[local_color_index_order]
            # local_cmap = np.array([colors(1.0)
            #                    for colors in drift_cmap])[local_color_index_order]

        if image.shape[0] == 3:
            dsize = image.shape[1:3]  # pytorch
        else:
            dsize = image.shape[0:2]  # tf
        BaseCraft._show(image, **plot_kwargs)

        image_u = self.transform(image)[0]
        for i, c_id in enumerate(most_important_concepts[::-1]):
            heatmap = image_u[:, :, c_id]

            # only show concept if excess N-th percentile
            sigma = np.percentile(np.array(heatmap).flatten(), filter_percentile)
            heatmap = heatmap * np.array(heatmap > sigma, np.float32)

            # resize the heatmap before cliping
            heatmap = cv2.resize(heatmap[:, :, None], dsize=dsize,
                                 interpolation=cv2.INTER_CUBIC)
            if clip_percentile:
                heatmap = _clip_percentile(heatmap, clip_percentile)

            BaseCraft._show(heatmap, cmap=local_cmap[::-1][i], alpha=alpha, **plot_kwargs)

    def plot_concepts_importances(self,
                                  class_concepts: int,
                                  importances: np.ndarray = None,
                                  display_importance_order: DisplayImportancesOrder = \
                                          DisplayImportancesOrder.GLOBAL,
                                  nb_most_important_concepts: int = None,
                                  verbose: bool = False):
        """
        Plot a bar chart displaying the importance value of each concept.

        Parameters
        ----------
        importances
            The importances computed by the estimate_importance() method.
            Default is None, in this case the importances computed on the whole
            dataset will be used.
        display_importance_order
            Selects the order in which the concepts will be displayed, either following the
            global importance on the whole dataset (same order for all images) or the local
            importance of the concepts for a single image sample (local importance).
        nb_most_important_concepts
            The number of concepts to display. If None is provided, then all the concepts
            will be displayed unordered, otherwise only nb_most_important_concepts will be
            displayed, ordered by importance.
            Default is None.
        verbose
            If True, then print the importance value of each concept, otherwise no textual
            output will be printed.
        """

        if importances is None:
            # global
            importances = self.sensitivities[class_concepts].importances
            most_important_concepts = self.sensitivities[class_concepts].most_important_concepts
        else:
            # local
            most_important_concepts = np.argsort(importances)[::-1]

        if nb_most_important_concepts is None:
            # display all concepts not ordered
            global_color_index_order = self.sensitivities[class_concepts].most_important_concepts
            local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                       for local_c in range(len(importances))]
            colors = np.array([colors(1.0)
                               for colors in self.sensitivities[class_concepts].cmaps])[local_color_index_order]

            plt.bar(range(len(importances)), importances, color=colors)
            plt.xticks(range(len(importances)))

        else:
            # only display the nb_most_important_concepts concepts in their importance order
            most_important_concepts = most_important_concepts[:nb_most_important_concepts]

            # Find the correct color index
            global_color_index_order = np.argsort(self.sensitivities[class_concepts].importances)[::-1]
            local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                       for local_c in most_important_concepts]
            if class_concepts == 0:
                color_tup = [plt.get_cmap('tab20c').colors[0], plt.get_cmap('tab20c').colors[1],plt.get_cmap('tab20c').colors[2],
                             plt.get_cmap('tab20c').colors[3], plt.get_cmap('tab20').colors[19]]
                for i in range(25):
                    color_tup.append((plt.get_cmap('tab20c').colors + plt.get_cmap('Set3').colors)[i])

                drift_cmap = [Sensitivity._get_alpha_cmap(cmap) for cmap in color_tup]

                colors = np.array([colors(1.0)
                               for colors in drift_cmap])[local_color_index_order]
            if class_concepts == 1:
                color_tup = [plt.get_cmap('tab20b').colors[12], plt.get_cmap('tab20b').colors[13],
                             plt.get_cmap('tab20b').colors[14],
                             plt.get_cmap('tab20b').colors[15], plt.get_cmap('tab20').colors[6]]
                for i in range(25):
                    color_tup.append((plt.get_cmap('tab20c').colors + plt.get_cmap('Set3').colors)[i])

                drift_cmap = [Sensitivity._get_alpha_cmap(cmap) for cmap in color_tup]

                colors = np.array([colors(1.0)
                                   for colors in drift_cmap])[local_color_index_order]


            plt.bar(range(len(importances[most_important_concepts])),
                    importances[most_important_concepts], color=colors)
            plt.xticks(ticks=range(len(most_important_concepts)),
                       labels=most_important_concepts)

        if display_importance_order == DisplayImportancesOrder.GLOBAL:
            importance_order = "Global"
        else:
            importance_order = "Local"
        plt.title(f"{importance_order} Concept Importance")

        if verbose:
            for c_id in most_important_concepts:
                print(f"Concept {c_id} has an importance value of {importances[c_id]:.2f}")

