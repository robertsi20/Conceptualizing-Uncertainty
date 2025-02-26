from DeepView.deepview.DeepView import DeepViewSelector
from DeepView.deepview.fisher_metric import calculate_fisher
from DeepView.deepview.Selector import SelectFromCollection
# from xplique.concepts import CraftTorch as Craft
from xplique.attributions.global_sensitivity_analysis import (HaltonSequenceRS, JansenEstimator)
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import torch
#
# class DeepViewConcept(DeepViewSelector):
#
#     def __init__(self, *args, class_dict=None, craft_obj, n_concepts, h,g, **kwargs):
#         super().__init__(*args, class_dict, **kwargs)
#         self.craft = craft_obj
#         self.concept_vectors = np.empty([0, 8, 8, n_concepts])
#         self.imp_concept_vectors = np.empty([0,n_concepts])
#         self.class_dict = class_dict
#         self.concept_distances = np.array([])
#         self.n_concepts = n_concepts
#         self.imp_concepts = np.array([])
#         self.h = h
#         self.g = g
#
#     @property
#     def concept_distance(self):
#         '''
#         Returns the distance between the concept vectors of the input.
#         :return:
#         '''
#         return self.concept_distances
#
#
#     def get_imp_concepts(self, samples):
#         """
#         Returns the concept vectors of the input.TODO: check if the importance function is also doing
#         something else
#         :param samples:
#         :return:
#         """
#         # print(samples.shape)
#         # print(samples[0].shape)
#         imp_concepts = np.array([self.craft.estimate_importance(image) for image in samples])
#         return imp_concepts
#
#     def mesh_importance(self, inputs: np.ndarray = None, nb_design: int = 32) -> np.ndarray:
#         """
#         Code taken from https://github.com/deel-ai/xplique/blob/master/xplique/concepts/craft.py estimate importance function
#
#         Estimates the importance of each concept for a given class, either globally
#         on the whole dataset provided in the fit() method (in this case, inputs shall
#         be set to None), or locally on a specific input image.
#
#         Parameters
#         ----------
#         inputs : numpy array or Tensor
#             The input data on which to compute the importances.
#             If None, then the inputs provided in the fit() method
#             will be used (global importance of the whole dataset).
#             Default is None.
#         nb_design
#             The number of design to use for the importance estimation. Default is 32.
#
#         Returns
#         -------
#         importances
#             The Sobol total index (importance score) for each concept.
#
#         """
#         self.craft.check_if_fitted()
#
#         compute_global_importances = False
#         if inputs is None:
#             inputs = self.craft.factorization.inputs
#             compute_global_importances = True
#
#         # coeffs_u = self.transform(inputs)
#         coeffs_u = inputs
#         # print(len(coeffs_u.shape))
#         masks = HaltonSequenceRS()(self.craft.number_of_concepts, nb_design=nb_design)
#         estimator = JansenEstimator()
#
#         importances = []
#
#         if len(coeffs_u.shape) == 2:
#             # apply the original method of the paper
#
#             for coeff in coeffs_u:
#                 u_perturbated = coeff[None, :] * masks
#                 a_perturbated = u_perturbated @ self.craft.factorization.concept_bank_w
#
#                 y_pred = self.craft._logit_predict(a_perturbated)
#                 y_pred = y_pred[:, self.craft.factorization.class_id]
#
#                 stis = estimator(masks, y_pred, nb_design)
#
#                 importances.append(stis)
#
#         elif len(coeffs_u.shape) == 4:
#             # apply a re-parameterization trick and use mask on all localization for a given
#             # concept id to estimate sobol indices
#             for coeff in coeffs_u:
#                 u_perturbated = coeff[None, :] * masks[:, None, None, :]
#                 a_perturbated = np.reshape(u_perturbated,
#                                            (-1, coeff.shape[-1])) @ self.craft.factorization.concept_bank_w
#                 a_perturbated = np.reshape(a_perturbated,
#                                            (len(masks), coeffs_u.shape[1], coeffs_u.shape[2], -1))
#
#                 # a_perturbated: (N, H, W, C)
#                 y_pred = self.craft._logit_predict(a_perturbated)
#                 y_pred = y_pred[:, self.craft.factorization.class_id]
#
#                 stis = estimator(masks, y_pred, nb_design)
#
#                 importances.append(stis)
#
#         # print(np.array(importances).shape)
#         importances = np.mean(importances, 0)
#
#
#         return importances
#
#     def compute_grid(self):
#         '''
#         Computes the visualisation of the decision boundaries.
#         '''
#         if self.verbose:
#             print('Computing decision regions ...')
#         # get extent of embedding
#         x_min, y_min, x_max, y_max = self._get_plot_measures()
#         # create grid
#         xs = np.linspace(x_min, x_max, self.resolution)
#         ys = np.linspace(y_min, y_max, self.resolution)
#         self.grid = np.array(np.meshgrid(xs, ys))
#         grid = np.swapaxes(self.grid.reshape(self.grid.shape[0],-1),0,1)
#
#         # map gridmpoint to images
#         grid_samples = self.inverse(grid)
#
#         # print(grid_samples.shape)
#         # print(grid_samples[0].shape)
#         # mesh_preds = np.array([self.mesh_importance(np.array(grid_samp, dtype=np.float32).reshape((1,8,8,10))) for grid_samp in grid_samples])
#         mesh_preds = self.h(grid_samples)
#         mesh_preds = mesh_preds + 1e-8
#         # print(mesh_preds.shape)
#
#         self.mesh_classes = mesh_preds.argmax(axis=1)
#         # mesh_max_class = 9
#         mesh_max_class = max(self.mesh_classes)
#
#         # get color of gridpoints
#         color = self.cmap(self.mesh_classes/mesh_max_class)
#
#         # scale colors by certainty
#         h = -(mesh_preds*np.log(mesh_preds)).sum(axis=1)/np.log(self.n_classes)
#         h = (h/h.max()).reshape(-1, 1)
#         # adjust brightness
#         h = np.clip(h*1.2, 0, 1)
#         color = color[:,0:3]
#         color = (1-h)*(0.5*color) + h*np.ones(color.shape, dtype=np.uint8)
#         decision_view = color.reshape(self.resolution, self.resolution, 3)
#         return decision_view
#
#     def update_mappings(self):
#         if self.verbose:
#             print('Embedding samples ...')
#
#         self.mapper.fit(self.concept_distances)
#         self.embedded = self.mapper.transform(self.concept_distances)
#         # activations = self.g(self.samples)
#         # self.inverse.fit(self.embedded, activations)
#         # self.classifier_view = self.compute_grid()
#         # self.background_at = np.array([self.get_mesh_prediction_at(x, y) for x, y in self.embedded])
#
#     def queue_samples(self, samples, labels, preds, imp_concept_vectors, important_concept,concept_vectors):
#         '''
#         Adds samples labels and predictions to the according lists of
#         this deepview object. Old values will be discarded, when there are
#         more then max_samples.
#         '''
#         # add new samples and remove depricated samples
#         self.samples = np.concatenate((samples, self.samples))[:self.max_samples]
#         self.y_pred = np.concatenate((preds, self.y_pred))[:self.max_samples]
#         self.y_true = np.concatenate((labels, self.y_true))[:self.max_samples]
#         self.imp_concept_vectors = np.concatenate((imp_concept_vectors, self.imp_concept_vectors))[:self.max_samples]
#         self.concept_vectors = np.concatenate((concept_vectors, self.concept_vectors))[:self.max_samples]
#         self.imp_concepts = np.concatenate((important_concept, self.imp_concepts))[:self.max_samples]
#
#     def add_samples(self, samples, labels):
#         '''
#         Adds samples points to the visualization.
#
#         Parameters
#         ----------
#         samples : array-like
#             List of new sample points [n_samples, *data_shape]
#         labels : array-like
#             List of labels for the sample points [n_samples, 1]
#         '''
#         # get predictions for the new samples
#         Y_probs = self._predict_batches(samples)
#         Y_preds = Y_probs.argmax(axis=1)
#         concept_vectors = self.craft.transform(samples)
#         # test = self.craft.transform(samples[0])
#         # print(test.shape)
#         # print(concept_vectors.shape)
#         imp_concept_vectors = self.get_imp_concepts(samples)
#         imp_concept = np.argmax(imp_concept_vectors, axis=1)
#
#         # add new values to the DeepView lists
#         self.queue_samples(samples, labels, Y_preds, imp_concept_vectors, imp_concept, concept_vectors)
#
#         # calculate new distances
#         # keep for when we decide to intregate different classes
#         # new_discr, new_eucl = calculate_fisher(self.model, concept_vectors, self.concept_vectors,
#         #                                        self.n, self.batch_size, self.n_classes, self.metric, False,
#         #                                        self.verbose)
#         # calculate new concept distances
#         new_discr, new_concept = calculate_fisher(self.model, imp_concept_vectors, self.imp_concept_vectors,
#             self.n, self.batch_size, self.n_classes, self.metric, False, self.verbose)
#
#
#
#         # add new distances
#         # self.discr_distances = self.update_matrix(self.discr_distances, new_discr)
#         # self.eucl_distances = self.update_matrix(self.eucl_distances, new_eucl)
#
#         self.concept_distances = self.update_matrix(self.concept_distances, new_concept)
#
#         # update mappings
#         self.update_mappings()
#
#     def _init_plots(self):
#         '''
#         Initialises matplotlib artists and plots.
#         '''
#         if self.interactive:
#             plt.ion()
#         self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
#         self.ax.set_title(self.title)
#         self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
#         self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
#                                        interpolation='gaussian', zorder=0, vmin=0, vmax=1)
#
#         self.sample_plots = []
#         # global_color_index_order = np.argsort(self.craft.sensitivity.importances)[::-1]
#         # local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
#         #                            for local_c in self.craft.sensitivity.most_important_concepts]
#         # local_cmap = np.array(self.craft.sensitivity.cmaps)[local_color_index_order]
#         #
#         # for i,c in enumerate(global_color_index_order):
#         #     color = local_cmap[i]
#         #     plot = self.ax.plot([], [], 'o', label=str(c),
#         #                         color=color(1.0), zorder=2, picker=mpl.rcParams['lines.markersize'])
#         #     self.sample_plots.append(plot[0])
#
#         # for c in range(self.n_classes):
#         #     color = self.cmap(c / (self.n_classes - 1))
#         #     plot = self.ax.plot([], [], 'o', markeredgecolor=color,
#         #                         fillstyle='none', ms=12, mew=2.5, zorder=1)
#         #     self.sample_plots.append(plot[0])
#
#         # set the mouse-event listeners
#         self.fig.canvas.mpl_connect('key_press_event', self.show_sample)
#         self.disable_synth = False
#         self.ax.set_axis_off()
#         self.ax.legend()
#
#     def get_artist_sample(self, sample_ids):
#         """Maps the location of embedded points to their image.
#
#         Parameters
#         ----------
#         sample_ids: tuple
#             The ids of selected vertices.
#         """
#         # sample_id = np.argmin(np.linalg.norm(self.embedded - point, axis=1))
#         yps = []
#         yts = []
#         artist_concs = []
#         for sample_id in sample_ids:
#             yp, yt  = (int(self.y_pred[sample_id]), int(self.y_true[sample_id]))
#             yps.append(yp)
#             yts.append(yt)
#             artist_concs.append(self.imp_concept_vectors[sample_id])
#         return sample_ids, yps, yts, artist_concs
#
#     def show(self):
#         '''
#         Shows the current plot.
#         '''
#         if not hasattr(self, 'fig'):
#             self._init_plots()
#
#         x_min, y_min, x_max, y_max = self._get_plot_measures()
#
#         # self.cls_plot.set_data(self.classifier_view)
#         # self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
#         self.ax.set_xlim((x_min, x_max))
#         self.ax.set_ylim((y_min, y_max))
#
#         params_str = 'batch size: %d - n: %d - $\lambda$: %.2f - res: %d'
#         desc = params_str % (self.batch_size, self.n, self.lam, self.resolution)
#         self.desc.set_text(desc)
#         scatter_plots = []
#         labels = []
#
#         global_color_index_order = np.argsort(self.craft.sensitivity.importances)[::-1]
#         local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
#                                    for local_c in self.craft.sensitivity.most_important_concepts]
#         local_cmap = np.array(self.craft.sensitivity.cmaps)[local_color_index_order]
#
#         # ordered_concepts = np.argsort(self.craft.sensitivity.importances)[::-1]
#         unique_labels = np.sort(np.unique(self.y_true))
#         for ti in unique_labels:
#             t_fil = self.y_true == ti
#             for i, c in enumerate(global_color_index_order):
#                 data = self.embedded[t_fil][self.imp_concepts[t_fil] == c]
#                 color = local_cmap[i]
#                 if ti == unique_labels[0]:
#                     scatter = self.ax.scatter(data[:, 0], data[:, 1], color=color(1.0))
#
#                     # Only add to legend for the first group to avoid duplicates
#                     scatter_plots.append(scatter)
#                     labels.append(c)
#                 else:
#                     scatter = self.ax.scatter(data[:, 0], data[:, 1], marker=(5, 2), color=color(1.0))
#
#             # Create legend for classes
#             legend1 = self.ax.legend(scatter_plots, labels, loc="upper left",
#                                      bbox_to_anchor=(1.04, 1), title="Concepts")
#             self.ax.add_artist(legend1)
#
#             # Create dummy scatter plots for the gender legend
#             g1_scatter = self.ax.scatter([], [], marker='o', color='black', label='Beagle')
#             g2_scatter = self.ax.scatter([], [], marker=(5, 2), color='black', label='English Foxhound')
#
#             # Create legend for gender
#             legend2 = self.ax.legend(handles=[g1_scatter, g2_scatter], loc="upper right", title="Classes")
#             self.ax.add_artist(legend2)
#
#             # Adjust subplot parameters to make room for the legends
#             plt.subplots_adjust(right=0.45)
#
#         # for c in range(self.n_concepts):
#         #     data = self.embedded[np.logical_and(self.y_pred==c, self.background_at!=c)]
#         # self.sample_plots[self.n_classes+c].set_data(data.transpose())
#
#         if os.name == 'posix':
#             self.fig.canvas.manager.window.raise_()
#
#         self.selector = SelectFromCollection(self.ax, self.embedded)
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()
#         plt.show()
#
#     def show_sample(self, event):
#         '''
#         Invoked when the user clicks on the plot. Determines the
#         embedded or synthesised sample at the click location and
#         passes it to the data_viz method, together with the prediction,
#         if present a groun truth label and the 2D click location.
#         '''
#
#         # when there is an artist attribute, a
#         # concrete sample was clicked, otherwise
#         # show the according synthesised image
#
#         if event.key == "enter":
#             indices = self.selector.ind
#             sample, p, t, conc = self.get_artist_sample(indices)
#             # title = '%s <-> %s' if p != t else '%s --- %s'
#             # title = title % (self.classes[p], self.classes[t])
#             self.disable_synth = True
#         elif not self.disable_synth:
#             # workaraound: inverse embedding needs more points
#             # otherwise it doens't work --> [point]*5
#             point = np.array([[event.xdata, event.ydata]] * 5)
#
#             # if the outside of the plot was clicked, points are None
#             if None in point[0]:
#                 return
#
#             sample = self.inverse(point)[0]
#             sample += abs(sample.min())
#             sample /= sample.max()
#             # title = 'Synthesised at [%.1f, %.1f]' % tuple(point[0])
#             p, t = self.get_mesh_prediction_at(*point[0]), None
#         else:
#             self.disable_synth = False
#             return
#
#         if self.data_viz is not None:
#             self.data_viz(sample, p, t, conc, self.cmap)
#             return
#         else:
#             warnings.warn("Data visualization not possible, as the partnet_grasp points have"
#                           "no image shape. Pass a function in the data_viz argument,"
#                           "to enable custom partnet_grasp visualization.")
#             return


# class DeepViewConcept

class DeepViewConcept(DeepViewSelector):

    def __init__(self, *args, class_dict=None, craft_obj, n_concepts, h, **kwargs):
        super().__init__(*args, class_dict, **kwargs)
        self.craft = craft_obj
        self.concept_vectors = np.empty([0, 8, 8, n_concepts])
        self.imp_concept_vectors = np.empty([0,n_concepts])
        self.class_dict = class_dict
        self.concept_distances = np.array([])
        self.n_concepts = n_concepts
        self.imp_concepts = np.array([])
        self.class_cmap = plt.get_cmap("tab10")
        self.h = h
        # self.g = g

    @property
    def concept_distance(self):
        '''
        Returns the distance between the concept vectors of the input.
        :return:
        '''
        return self.concept_distances


    def get_imp_concepts(self, samples, y_preds):
        """
        Returns the concept vectors of the input.TODO: check if the importance function is also doing
        something else
        :param samples:
        :return:
        """
        # print(samples.shape)
        # print(samples[0].shape)
        imp_concepts = np.array([self.craft.estimate_importance_helper(image, labels=y_preds[i], class_of_interest=y_preds[i])
                                 for (i,image) in enumerate(samples)])
        return imp_concepts

    def mesh_importance(self, inputs: np.ndarray = None, nb_design: int = 32) -> np.ndarray:
        """
        Code taken from https://github.com/deel-ai/xplique/blob/master/xplique/concepts/craft.py estimate importance function

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
        self.craft.check_if_fitted()

        compute_global_importances = False
        if inputs is None:
            inputs = self.craft.factorization.inputs
            compute_global_importances = True

        # coeffs_u = self.transform(inputs)
        coeffs_u = inputs
        # print(len(coeffs_u.shape))
        masks = HaltonSequenceRS()(self.craft.number_of_concepts, nb_design=nb_design)
        estimator = JansenEstimator()

        importances = []

        if len(coeffs_u.shape) == 2:
            # apply the original method of the paper

            for coeff in coeffs_u:
                u_perturbated = coeff[None, :] * masks
                a_perturbated = u_perturbated @ self.craft.factorization.concept_bank_w

                y_pred = self.craft._logit_predict(a_perturbated)
                y_pred = y_pred[:, self.craft.factorization.class_id]

                stis = estimator(masks, y_pred, nb_design)

                importances.append(stis)

        elif len(coeffs_u.shape) == 4:
            # apply a re-parameterization trick and use mask on all localization for a given
            # concept id to estimate sobol indices
            for coeff in coeffs_u:
                u_perturbated = coeff[None, :] * masks[:, None, None, :]
                a_perturbated = np.reshape(u_perturbated,
                                           (-1, coeff.shape[-1])) @ self.craft.factorization.concept_bank_w
                a_perturbated = np.reshape(a_perturbated,
                                           (len(masks), coeffs_u.shape[1], coeffs_u.shape[2], -1))

                # a_perturbated: (N, H, W, C)
                y_pred = self.craft._logit_predict(a_perturbated)
                y_pred = y_pred[:, self.craft.factorization.class_id]

                stis = estimator(masks, y_pred, nb_design)

                importances.append(stis)

        # print(np.array(importances).shape)
        importances = np.mean(importances, 0)


        return importances

    def compute_grid(self):
        '''
        Computes the visualisation of the decision boundaries.
        '''
        if self.verbose:
            print('Computing decision regions ...')
        # get extent of embedding
        x_min, y_min, x_max, y_max = self._get_plot_measures()
        # create grid
        xs = np.linspace(x_min, x_max, self.resolution)
        ys = np.linspace(y_min, y_max, self.resolution)
        self.grid = np.array(np.meshgrid(xs, ys))
        grid = np.swapaxes(self.grid.reshape(self.grid.shape[0],-1),0,1)

        # map gridmpoint to images
        grid_samples = self.inverse(grid)
        print(grid_samples.shape)

        # print(grid_samples.shape)
        # print(grid_samples[0].shape)
        # mesh_preds = np.array([self.mesh_importance(np.array(grid_samp, dtype=np.float32).reshape((1,8,8,10))) for grid_samp in grid_samples])
        grid_samples = grid_samples.astype(np.float32) @ self.craft.factorization.concept_bank_w
        mesh_preds = self.craft._logit_predict(grid_samples)
        mesh_preds = mesh_preds + 1e-8
        # print(mesh_preds.shape)

        #since the current predictor goes to logits space
        # and has 1000 classes, need to implement this workaround
        # beagle_mesh_preds = mesh_preds[:, 162]
        # hound_mesh_preds = mesh_preds[:, 167]
        # mesh_logits = np.concatenate((beagle_mesh_preds, hound_mesh_preds)).reshape((639, 2))
        # mesh_preds = np.exp(mesh_logits) / sum(np.exp(mesh_logits))

        self.mesh_classes = mesh_preds.argmax(axis=1)
        # mesh_max_class = 9
        mesh_max_class = max(self.mesh_classes)

        # get color of gridpoints
        color = self.class_cmap(self.mesh_classes/mesh_max_class)

        # scale colors by certainty
        h = -(mesh_preds*np.log(mesh_preds)).sum(axis=1)/np.log(mesh_preds.shape[1])
        h = (h/h.max()).reshape(-1, 1)
        # adjust brightness
        h = np.clip(h*1.2, 0, 1)
        color = color[:,0:3]
        color = (1-h)*(0.5*color) + h*np.ones(color.shape, dtype=np.uint8)
        decision_view = color.reshape(self.resolution, self.resolution, 3)
        return decision_view

    def update_mappings(self):
        if self.verbose:
            print('Embedding samples ...')

        self.mapper.fit(self.concept_distances)
        self.embedded = self.mapper.transform(self.concept_distances)
        # activations = self.g(self.samples)
        print(self.concept_vectors.shape)
        self.inverse.fit(self.embedded, self.concept_vectors)
        self.classifier_view = self.compute_grid()
        # self.background_at = np.array([self.get_mesh_prediction_at(x, y) for x, y in self.embedded])

    def queue_samples(self, samples, labels, preds, imp_concept_vectors, important_concept,concept_vectors):
        '''
        Adds samples labels and predictions to the according lists of
        this deepview object. Old values will be discarded, when there are
        more then max_samples.
        '''
        # add new samples and remove depricated samples
        self.samples = np.concatenate((samples, self.samples))[:self.max_samples]
        self.y_pred = np.concatenate((preds, self.y_pred))[:self.max_samples]
        self.y_true = np.concatenate((labels, self.y_true))[:self.max_samples]
        self.imp_concept_vectors = np.concatenate((imp_concept_vectors, self.imp_concept_vectors))[:self.max_samples]
        self.concept_vectors = np.concatenate((concept_vectors, self.concept_vectors))[:self.max_samples]
        self.imp_concepts = np.concatenate((important_concept, self.imp_concepts))[:self.max_samples]

    def add_samples(self, samples, labels):
        '''
        Adds samples points to the visualization.

        Parameters
        ----------
        samples : array-like
            List of new sample points [n_samples, *data_shape]
        labels : array-like
            List of labels for the sample points [n_samples, 1]
        '''
        # get predictions for the new samples
        Y_probs = self._predict_batches(samples)
        Y_preds = Y_probs.argmax(axis=1)
        concept_vectors = self.craft.transform(samples, Y_preds)
        print(concept_vectors.shape)
        # test = self.craft.transform(samples[0])
        # print(test.shape)
        # print(concept_vectors.shape)
        imp_concept_vectors = self.get_imp_concepts(samples, Y_preds)
        imp_concept = np.argmax(imp_concept_vectors, axis=1)

        # add new values to the DeepView lists
        self.queue_samples(samples, labels, Y_preds, imp_concept_vectors, imp_concept, concept_vectors)

        # calculate new distances
        # keep for when we decide to intregate different classes
        # new_discr, new_eucl = calculate_fisher(self.model, concept_vectors, self.concept_vectors,
        #                                        self.n, self.batch_size, self.n_classes, self.metric, False,
        #                                        self.verbose)
        # calculate new concept distances
        new_discr, new_concept = calculate_fisher(self.model, imp_concept_vectors, self.imp_concept_vectors,
            self.n, self.batch_size, self.n_classes, self.metric, self.disc_dist, self.verbose)



        # add new distances
        # self.discr_distances = self.update_matrix(self.discr_distances, new_discr)
        # self.eucl_distances = self.update_matrix(self.eucl_distances, new_eucl)

        self.concept_distances = self.update_matrix(self.concept_distances, new_concept)

        # update mappings
        self.update_mappings()

    def _init_plots(self):
        '''
        Initialises matplotlib artists and plots.
        '''
        if self.interactive:
            plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        self.ax.set_title(self.title)
        self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
                                       interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []
        # global_color_index_order = np.argsort(self.craft.sensitivity.importances)[::-1]
        # local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
        #                            for local_c in self.craft.sensitivity.most_important_concepts]
        # local_cmap = np.array(self.craft.sensitivity.cmaps)[local_color_index_order]
        #
        # for i,c in enumerate(global_color_index_order):
        #     color = local_cmap[i]
        #     plot = self.ax.plot([], [], 'o', label=str(c),
        #                         color=color(1.0), zorder=2, picker=mpl.rcParams['lines.markersize'])
        #     self.sample_plots.append(plot[0])

        # for c in range(self.n_classes):
        #     color = self.cmap(c / (self.n_classes - 1))
        #     plot = self.ax.plot([], [], 'o', markeredgecolor=color,
        #                         fillstyle='none', ms=12, mew=2.5, zorder=1)
        #     self.sample_plots.append(plot[0])

        # set the mouse-event listeners
        self.fig.canvas.mpl_connect('key_press_event', self.show_sample)
        self.disable_synth = False
        self.ax.set_axis_off()
        self.ax.legend()

    def get_artist_sample(self, sample_ids):
        """Maps the location of embedded points to their image.

        Parameters
        ----------
        sample_ids: tuple
            The ids of selected vertices.
        """
        # sample_id = np.argmin(np.linalg.norm(self.embedded - point, axis=1))
        yps = []
        yts = []
        artist_concs = []
        for sample_id in sample_ids:
            yp, yt  = (int(self.y_pred[sample_id]), int(self.y_true[sample_id]))
            yps.append(yp)
            yts.append(yt)
            artist_concs.append(self.imp_concept_vectors[sample_id])
        return sample_ids, yps, yts, artist_concs

    def show(self):
        '''
        Shows the current plot.
        '''
        if not hasattr(self, 'fig'):
            self._init_plots()

        x_min, y_min, x_max, y_max = self._get_plot_measures()

        self.cls_plot.set_data(self.classifier_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'batch size: %d - n: %d - $\lambda$: %.2f - res: %d'
        desc = params_str % (self.batch_size, self.n, self.lam, self.resolution)
        self.desc.set_text(desc)
        scatter_plots = []
        labels = []

        global_color_index_order = np.argsort(self.craft.sensitivities[0].importances)[::-1]
        local_color_index_order = [np.where(global_color_index_order == local_c)[0][0]
                                   for local_c in self.craft.sensitivities[0].most_important_concepts]
        local_cmap = np.array(self.craft.sensitivities[0].cmaps)[local_color_index_order]

        # ordered_concepts = np.argsort(self.craft.sensitivity.importances)[::-1]
        unique_labels = np.sort(np.unique(self.y_true))
        for ti in unique_labels:
            t_fil = self.y_true == ti
            for i, c in enumerate(global_color_index_order):
                data = self.embedded[t_fil][self.imp_concepts[t_fil] == c]
                color = local_cmap[i]
                if ti == unique_labels[0]:
                    scatter = self.ax.scatter(data[:, 0], data[:, 1], color=color(1.0))

                    # Only add to legend for the first group to avoid duplicates
                    scatter_plots.append(scatter)
                    labels.append(c)
                else:
                    scatter = self.ax.scatter(data[:, 0], data[:, 1], marker=(5, 2), color=color(1.0))

            # Create legend for classes
            legend1 = self.ax.legend(scatter_plots, labels, loc="upper left",
                                     bbox_to_anchor=(1.04, 1), title="Concepts")
            self.ax.add_artist(legend1)

            # Create dummy scatter plots for the gender legend
            # g1_scatter = self.ax.scatter([], [], marker='o', color='black', label='Beagle')
            # g2_scatter = self.ax.scatter([], [], marker=(5, 2), color='black', label='English Foxhound')
            #
            # # Create legend for gender
            # legend2 = self.ax.legend(handles=[g1_scatter, g2_scatter], loc="upper right", title="Classes")
            # self.ax.add_artist(legend2)

            # Adjust subplot parameters to make room for the legends
            plt.subplots_adjust(right=0.45)

        # for c in range(self.n_concepts):
        #     data = self.embedded[np.logical_and(self.y_pred==c, self.background_at!=c)]
        # self.sample_plots[self.n_classes+c].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.selector = SelectFromCollection(self.ax, self.embedded)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

    def show_sample(self, event):
        '''
        Invoked when the user clicks on the plot. Determines the
        embedded or synthesised sample at the click location and
        passes it to the data_viz method, together with the prediction,
        if present a groun truth label and the 2D click location.
        '''

        # when there is an artist attribute, a
        # concrete sample was clicked, otherwise
        # show the according synthesised image

        if event.key == "enter":
            indices = self.selector.ind
            sample, p, t, conc = self.get_artist_sample(indices)
            # title = '%s <-> %s' if p != t else '%s --- %s'
            # title = title % (self.classes[p], self.classes[t])
            self.disable_synth = True
        elif not self.disable_synth:
            # workaraound: inverse embedding needs more points
            # otherwise it doens't work --> [point]*5
            point = np.array([[event.xdata, event.ydata]] * 5)

            # if the outside of the plot was clicked, points are None
            if None in point[0]:
                return

            sample = self.inverse(point)[0]
            sample += abs(sample.min())
            sample /= sample.max()
            # title = 'Synthesised at [%.1f, %.1f]' % tuple(point[0])
            p, t = self.get_mesh_prediction_at(*point[0]), None
        else:
            self.disable_synth = False
            return

        if self.data_viz is not None:
            self.data_viz(sample, p, t, conc, self.cmap)
            return
        else:
            warnings.warn("Data visualization not possible, as the partnet_grasp points have"
                          "no image shape. Pass a function in the data_viz argument,"
                          "to enable custom partnet_grasp visualization.")
            return
