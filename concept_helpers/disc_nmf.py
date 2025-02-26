import warnings

from sklearn import config_context
from sklearn.decomposition import NMF
from sklearn.decomposition._nmf import _beta_divergence, _special_sparse_dot, EPSILON, _beta_loss_to_float, \
    _fit_coordinate_descent, _check_init
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot

import scipy.sparse as sp
import time
from math import sqrt
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import check_non_negative, check_random_state, check_is_fitted
from sklearn.base import _fit_context
import numpy as np

import cProfile
import pstats



# def _multiplicative_update_w(
#                              X,
#                              W,
#                              H,
#                              beta_loss,
#                              l1_reg_W,
#                              l2_reg_W,
#                              gamma,
#                              H_sum=None,
#                              HHt=None,
#                              XHt=None,
#                              update_H=True,
#                              ):
#     """Update W in Multiplicative Update NMF."""
#     if beta_loss == 2:
#         # Numerator
#         if XHt is None:
#             XHt = safe_sparse_dot(X, H.T)
#         if update_H:
#             # avoid a copy of XHt, which will be re-computed (update_H=True)
#             numerator = XHt
#         else:
#             # preserve the XHt, which is not re-computed (update_H=False)
#             numerator = XHt.copy()
#
#         # Denominator
#         if HHt is None:
#             HHt = np.dot(H, H.T)
#         denominator = np.dot(W, HHt)
#
#     else:
#         # Numerator
#         # if X is sparse, compute WH only where X is non zero
#         WH_safe_X = _special_sparse_dot(W, H, X)
#         if sp.issparse(X):
#             WH_safe_X_data = WH_safe_X.data
#             X_data = X.data
#         else:
#             WH_safe_X_data = WH_safe_X
#             X_data = X
#             # copy used in the Denominator
#             WH = WH_safe_X.copy()
#             if beta_loss - 1.0 < 0:
#                 WH[WH < EPSILON] = EPSILON
#
#         # to avoid taking a negative power of zero
#         if beta_loss - 2.0 < 0:
#             WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON
#
#         if beta_loss == 1:
#             np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
#         elif beta_loss == 0:
#             # speeds up computation time
#             # refer to /numpy/numpy/issues/9363
#             WH_safe_X_data **= -1
#             WH_safe_X_data **= 2
#             # element-wise multiplication
#             WH_safe_X_data *= X_data
#         else:
#             WH_safe_X_data **= beta_loss - 2
#             # element-wise multiplication
#             WH_safe_X_data *= X_data
#
#         # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
#         numerator = safe_sparse_dot(WH_safe_X, H.T)
#
#         # Denominator
#         if beta_loss == 1:
#             if H_sum is None:
#                 H_sum = np.sum(H, axis=1)  # shape(n_components, )
#             denominator = H_sum[np.newaxis, :]
#
#         else:
#             # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
#             if sp.issparse(X):
#                 # memory efficient computation
#                 # (compute row by row, avoiding the dense matrix WH)
#                 WHHt = np.empty(W.shape)
#                 for i in range(X.shape[0]):
#                     WHi = np.dot(W[i, :], H)
#                     if beta_loss - 1 < 0:
#                         WHi[WHi < EPSILON] = EPSILON
#                     WHi **= beta_loss - 1
#                     WHHt[i, :] = np.dot(WHi, H.T)
#             else:
#                 WH **= beta_loss - 1
#                 WHHt = np.dot(WH, H.T)
#             denominator = WHHt
#
#     # Add L1 and L2 regularization
#     if l1_reg_W > 0:
#         denominator += l1_reg_W
#     if l2_reg_W > 0:
#         denominator = denominator + l2_reg_W * W
#     denominator[denominator == 0] = EPSILON
#
#     numerator /= denominator
#     delta_W = numerator
#
#     # gamma is in ]0, 1]
#     if gamma != 1:
#         delta_W **= gamma
#
#     W *= delta_W
#
#     return W, H_sum, HHt, XHt
#
#
# def _multiplicative_update_h(
#                              X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma, A=None, B=None, rho=None
#                              ):
#     """update H in Multiplicative Update NMF."""
#     if beta_loss == 2:
#         numerator = safe_sparse_dot(W.T, X)
#         denominator = np.linalg.multi_dot([W.T, W, H])
#
#     else:
#         # Numerator
#         WH_safe_X = _special_sparse_dot(W, H, X)
#         if sp.issparse(X):
#             WH_safe_X_data = WH_safe_X.data
#             X_data = X.data
#         else:
#             WH_safe_X_data = WH_safe_X
#             X_data = X
#             # copy used in the Denominator
#             WH = WH_safe_X.copy()
#             if beta_loss - 1.0 < 0:
#                 WH[WH < EPSILON] = EPSILON
#
#         # to avoid division by zero
#         if beta_loss - 2.0 < 0:
#             WH_safe_X_data[WH_safe_X_data < EPSILON] = EPSILON
#
#         if beta_loss == 1:
#             np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
#         elif beta_loss == 0:
#             # speeds up computation time
#             # refer to /numpy/numpy/issues/9363
#             WH_safe_X_data **= -1
#             WH_safe_X_data **= 2
#             # element-wise multiplication
#             WH_safe_X_data *= X_data
#         else:
#             WH_safe_X_data **= beta_loss - 2
#             # element-wise multiplication
#             WH_safe_X_data *= X_data
#
#         # here numerator = dot(W.T, (dot(W, H) ** (beta_loss - 2)) * X)
#         numerator = safe_sparse_dot(W.T, WH_safe_X)
#
#         # Denominator
#         if beta_loss == 1:
#             W_sum = np.sum(W, axis=0)  # shape(n_components, )
#             W_sum[W_sum == 0] = 1.0
#             denominator = W_sum[:, np.newaxis]
#
#         # beta_loss not in (1, 2)
#         else:
#             # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
#             if sp.issparse(X):
#                 # memory efficient computation
#                 # (compute column by column, avoiding the dense matrix WH)
#                 WtWH = np.empty(H.shape)
#                 for i in range(X.shape[1]):
#                     WHi = np.dot(W, H[:, i])
#                     if beta_loss - 1 < 0:
#                         WHi[WHi < EPSILON] = EPSILON
#                     WHi **= beta_loss - 1
#                     WtWH[:, i] = np.dot(W.T, WHi)
#             else:
#                 WH **= beta_loss - 1
#                 WtWH = np.dot(W.T, WH)
#             denominator = WtWH
#
#     # Add L1 and L2 regularization
#     if l1_reg_H > 0:
#         denominator += l1_reg_H
#     if l2_reg_H > 0:
#         denominator = denominator + l2_reg_H * H
#     denominator[denominator == 0] = EPSILON
#
#     if A is not None and B is not None:
#         # Updates for the online nmf
#         if gamma != 1:
#             H **= 1 / gamma
#         numerator *= H
#         A *= rho
#         B *= rho
#         A += numerator
#         B += denominator
#         H = A / B
#
#         if gamma != 1:
#             H **= gamma
#     else:
#         delta_H = numerator
#         delta_H /= denominator
#         if gamma != 1:
#             delta_H **= gamma
#         H *= delta_H
#
#     return H

def norm(x):
    """Dot product-based Euclidean norm implementation.

    See: http://fa.bianp.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm.
    """
    return sqrt(squared_norm(x))
def _initialize_nmf(X, n_components, init=None, eps=1e-6, random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Valid options:

        - None: 'nndsvda' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvda"
        else:
            init = "random"

    # Random initialization
    if init == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.standard_normal(size=(n_components, n_features)).astype(
            X.dtype, copy=False
        )
        W = avg * rng.standard_normal(size=(n_samples, n_components)).astype(
            X.dtype, copy=False
        )
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.standard_normal(size=len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.standard_normal(size=len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H

def _multiplicative_update_w(
                             X,
                             W,
                             H,
                             D,
                             beta_loss,
                             l1_reg_W,
                             l2_reg_W,
                             gamma,
                             H_sum=None,
                             HDHt=None,
                             XDHt=None,
                             update_H=True,
                             ):
    """Update W in Multiplicative Update NMF."""
    if beta_loss == 2:
        # Numerator
        if XDHt is None:
            # XHt = safe_sparse_dot(X, H.T)
            XDHt = np.linalg.multi_dot([X, D, H.T])
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XDHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            numerator = XDHt.copy()

        # Denominator
        if HDHt is None:
            # HHt = np.dot(H, H.T)
            HDHt = np.linalg.multi_dot([H, D, H.T])
        denominator = np.dot(W, HDHt)

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    W *= delta_W

    return W, H_sum, HDHt, XDHt


def _multiplicative_update_d(
                             X, W, H,D,I,p, beta_loss, l1_reg_H, l2_reg_H, gamma, A=None, B=None, rho=None
                             ):
    """update H in Multiplicative Update NMF."""

    Z = (X - np.dot(W, H))
    Di = np.sqrt(np.sum(Z ** 2, axis=0)) ** (2 - p)
    D = np.diag(p / np.maximum(2 * Di, 1e-10))

    return D

def _multiplicative_update_h(
                             X, W, H,D,I,mu, beta_loss, l1_reg_H, l2_reg_H, gamma, A=None, B=None, rho=None
                             ):
    """update H in Multiplicative Update NMF."""
    if beta_loss == 2:
        # numerator = safe_sparse_dot(W.T, X)
        numerator = np.linalg.multi_dot([W.T, X, D])
        # denominator = np.linalg.multi_dot([W.T, W, H,D])
        denom1 = np.linalg.multi_dot([W.T, W, H,D])
        # print(I.shape)
        # print(H.shape)
        denom2 = mu*(I*H)
        denominator = denom1 + denom2

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = EPSILON

    if A is not None and B is not None:
        # Updates for the online nmf
        if gamma != 1:
            H **= 1 / gamma
        numerator *= H
        A *= rho
        B *= rho
        A += numerator
        B += denominator
        H = A / B

        if gamma != 1:
            H **= gamma
    else:
        delta_H = numerator
        delta_H /= denominator
        if gamma != 1:
            delta_H **= gamma
        H *= delta_H

    return H


def normalize_matrices(U, V):
    """
    Normalize the matrix U and adjust the matrix V accordingly.

    Parameters:
    U (numpy.ndarray): The matrix to be normalized.
    V (numpy.ndarray): The matrix to be adjusted based on the norms of U.

    Returns:
    U_normalized (numpy.ndarray): The normalized U matrix.
    V_adjusted (numpy.ndarray): The adjusted V matrix.
    """
    # Step 1: Calculate the norms as the sum of the absolute values of U along axis 1
    norms = np.sum(U, axis=0)
    # print(norms.shape)

    # Step 2: Ensure the norms are not less than 1e-10
    norms = np.maximum(norms, 1e-10)

    tuy = norms[:, np.newaxis]

    nuy = norms[np.newaxis, :]
    # Step 3: Normalize U by dividing each row by its corresponding norm
    U_normalized = U / nuy
    # print(U_normalized.shape)

    # Step 4: Scale V by multiplying each row by the corresponding norm
    V_adjusted = V * tuy
    # print(V_adjusted.shape)

    return U_normalized, V_adjusted



def _fit_multiplicative_update(
                               X,
                               W,
                               H,
                               I,
                               D=None,
                               beta_loss="frobenius",
                               max_iter=200,
                               tol=1e-4,
                               l1_reg_W=0,
                               l1_reg_H=0,
                               l2_reg_W=0,
                               l2_reg_H=0,
                               update_H=True,
                               update_W=False,
                               p=1,
                               mu=.5,
                               normalize=True,
                               verbose=0,
                               ):
    """Compute Non-negative Matrix Factorization with Multiplicative Update.

    The objective function is _beta_divergence(X, WH) and is minimized with an
    alternating minimization of W and H. Each minimization is done with a
    Multiplicative Update.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant input matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros.

    max_iter : int, default=200
        Number of iterations.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    l1_reg_W : float, default=0.
        L1 regularization parameter for W.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_W : float, default=0.
        L2 regularization parameter for W.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : int, default=0
        The verbosity level.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Lee, D. D., & Seung, H., S. (2001). Algorithms for Non-negative Matrix
    Factorization. Adv. Neural Inform. Process. Syst.. 13.
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1.0 / (2.0 - beta_loss)
    elif beta_loss > 2:
        gamma = 1.0 / (beta_loss - 1.0)
    else:
        gamma = 1.0

    # used for the convergence criterion
    # print("X shape: ", X.shape)
    # print("W shape: ", W.shape)
    # print("H shape: ", H.shape)
    error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None
    D = np.eye(X.shape[1])

    if update_W:
        for n_iter in range(1, max_iter + 1):

            H = _multiplicative_update_h(
                X,
                W,
                H,
                D,
                I,
                mu=0,
                beta_loss=beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=gamma,
            )
            D = _multiplicative_update_d(
                X,
                W,
                H,
                D,
                I,
                p=p,
                beta_loss=beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=gamma, )

            if tol > 0 and n_iter % 10 == 0:
                error = _beta_divergence(X, W, H, beta_loss, square_root=True)

                if verbose:
                    iter_time = time.time()
                    print(
                        "Epoch %02d reached after %.3f seconds, error: %f"
                        % (n_iter, iter_time - start_time, error)
                    )

                if (previous_error - error) / error_at_init < tol:
                    break
                previous_error = error
            if normalize:
                _, H =  normalize_matrices(W,H)
            # _, H = normalize_matrices(W, H)

    else:
        D = np.eye(X.shape[1])
        for n_iter in range(1, max_iter + 1):
            # update W
            # H_sum, HHt and XHt are saved and reused if not update_H

            W, H_sum, HHt, XHt = _multiplicative_update_w(
                X,
                W,
                H,
                D,
                beta_loss=beta_loss,
                l1_reg_W=l1_reg_W,
                l2_reg_W=l2_reg_W,
                gamma=gamma,
                H_sum=H_sum,
                HDHt=HHt,
                XDHt=XHt,
                update_H=update_H,
            )

            # necessary for stability with beta_loss < 1
            if beta_loss < 1:
                W[W < np.finfo(np.float64).eps] = 0.0

            # update H (only at fit or fit_transform)
            if update_H:
                H = _multiplicative_update_h(
                    X,
                    W,
                    H,
                    D,
                    I,
                    mu=mu,
                    beta_loss=beta_loss,
                    l1_reg_H=l1_reg_H,
                    l2_reg_H=l2_reg_H,
                    gamma=gamma,
                )


                # These values will be recomputed since H changed
                H_sum, HHt, XHt = None, None, None

                # necessary for stability with beta_loss < 1
                if beta_loss <= 1:
                    H[H < np.finfo(np.float64).eps] = 0.0

            D = _multiplicative_update_d(
                X,
                W,
                H,
                D,
                I,
                p=p,
                beta_loss=beta_loss,
                l1_reg_H=l1_reg_H,
                l2_reg_H=l2_reg_H,
                gamma=gamma, )

            # print("W", W.shape)
            # print("H", H.shape)
            if normalize:
                W, H=  normalize_matrices(W,H)
            # W, H= normalize_matrices(W, H)

            # test convergence criterion every 10 iterations
            if tol > 0 and n_iter % 10 == 0:
                error = _beta_divergence(X, W, H, beta_loss, square_root=True)

                if verbose:
                    iter_time = time.time()
                    print(
                        "Epoch %02d reached after %.3f seconds, error: %f"
                        % (n_iter, iter_time - start_time, error)
                    )

                if (previous_error - error) / error_at_init < tol:
                    break
                previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print(
            "Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time)
        )
    # print("done")
    return W, H, n_iter, D, I
class RSNMF(NMF):

    def __init__(self, n_components=None,component_per_class=None, p=None, mu=None, normalize=True, init=None, solver='mu', beta_loss='frobenius',
                 tol=1e-4, max_iter=500, random_state=None, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0,
                 verbose=0, shuffle=False):

        # Call the constructor of the base class (NMF) with its parameters
        super().__init__(n_components=n_components, init=init, solver=solver,
                         beta_loss=beta_loss, tol=tol, max_iter=max_iter,
                         random_state=random_state, alpha_W=0.0, alpha_H=0.0, l1_ratio=l1_ratio,
                         verbose=verbose, shuffle=shuffle)
        self.p = p
        self.mu = mu
        self.normalize = normalize
        self.component_per_class = component_per_class

    def _check_w_h(self, X, W, H, update_H,update_W):
        """Check W and H, or initialize them."""
        n_samples, n_features = X.shape

        if self.init == "custom" and update_H:
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, self._n_components), "NMF (input W)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError(
                    "H and W should have the same dtype as X. Got "
                    "H.dtype = {} and W.dtype = {}.".format(H.dtype, W.dtype)
                )
        elif update_W:
            if self.solver == "mu":
                avg = np.sqrt(X.mean() / self._n_components)

                H = np.full((self._n_components, n_features), avg, dtype=X.dtype)
            else:

                H = np.zeros((self._n_components, n_features), dtype=X.dtype)

        elif not update_H and H is not None:
            if W is not None:
                warnings.warn(
                    "When update_H=False, the provided initial W is not used.",
                    RuntimeWarning,
                )

            # _check_init(H, (self._n_components, n_features), "NMF (input H)")
            if self._n_components == "auto":
                self._n_components = H.shape[0]

            if H.dtype != X.dtype:
                raise TypeError(
                    "H should have the same dtype as X. Got H.dtype = {}.".format(
                        H.dtype
                    )
                )

            # 'mu' solver should not be initialized by zeros
            if self.solver == "mu":
                avg = np.sqrt(X.mean() / self._n_components)

                W = np.full((n_samples,self._n_components), avg, dtype=X.dtype)
            else:

                W = np.zeros((n_samples, self._n_components), dtype=X.dtype)


        else:
            if W is not None or H is not None:
                warnings.warn(
                    (
                        "When init!='custom', provided W or H are ignored. Set "
                        " init='custom' to use them as initialization."
                    ),
                    RuntimeWarning,
                )

            if self._n_components == "auto":
                self._n_components = X.shape[1]

            W, H = _initialize_nmf(
                X, self._n_components, init=self.init, random_state=self.random_state
            )
        return W, H
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

            This is more efficient than calling fit followed by transform.

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training vector, where `n_samples` is the number of samples
                and `n_features` is the number of features.

            y : Ignored
                Not used, present for API consistency by convention.

            W : array-like of shape (n_samples, n_components), default=None
                If `init='custom'`, it is used as initial guess for the solution.
                If `None`, uses the initialisation method specified in `init`.

            H : array-like of shape (n_components, n_features), default=None
                If `init='custom'`, it is used as initial guess for the solution.
                If `None`, uses the initialisation method specified in `init`.

            Returns
            -------
            W : ndarray of shape (n_samples, n_components)
                Transformed data.
        """
        X = self._validate_data(
                X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
            )

        with config_context(assume_finite=True):
                W, H, n_iter, D, I = self._fit_transform(X, y, W=W, H=H)

        self.reconstruction_err_ = _beta_divergence(
                X, W, H, self._beta_loss, square_root=True
            )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        self.W = W
        self.D = D
        self.I = I

        return W, H

    def _fit_transform(self, X, y=None, W=None, H=None,I=None,D=None, update_H=True, update_W=False):
        """Learn a NMF model for the data X and returns the transformed data.

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Data matrix to be decomposed

            y : Ignored

            W : array-like of shape (n_samples, n_components), default=None
                If `init='custom'`, it is used as initial guess for the solution.
                If `update_H=False`, it is initialised as an array of zeros, unless
                `solver='mu'`, then it is filled with values calculated by
                `np.sqrt(X.mean() / self._n_components)`.
                If `None`, uses the initialisation method specified in `init`.

            H : array-like of shape (n_components, n_features), default=None
                If `init='custom'`, it is used as initial guess for the solution.
                If `update_H=False`, it is used as a constant, to solve for W only.
                If `None`, uses the initialisation method specified in `init`.

            update_H : bool, default=True
                If True, both W and H will be estimated from initial guesses,
                this corresponds to a call to the 'fit_transform' method.
                If False, only W will be estimated, this corresponds to a call
                to the 'transform' method.

            Returns
            -------
            W : ndarray of shape (n_samples, n_components)
                Transformed data.

            H : ndarray of shape (n_components, n_features)
                Factorization matrix, sometimes called 'dictionary'.

            n_iter_ : int
                Actual number of iterations.
        """
        check_non_negative(X, "NMF (input X)")

            # check parameters
        self._check_params(X)

        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError(
                    "When beta_loss <= 0 and X contains zeros, "
                    "the solver may diverge. Please add small values "
                    "to X, or use a positive beta_loss."
                )

            # initialize or check W and H
        # self.num_classes = len(np.unique(y))
        I = y
        # if I is None:
        #     I = init_I(self.component_per_class, X.shape[1],y)
        #     # print(I)
        #     # print(I.shape)


        W, H = self._check_w_h(X, W, H, update_H, update_W)

            # scale the regularization terms
        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)

        if self.solver == "cd":
            W, H, n_iter = _fit_coordinate_descent(
                    X,
                    W,
                    H,
                    self.tol,
                    self.max_iter,
                    l1_reg_W,
                    l1_reg_H,
                    l2_reg_W,
                    l2_reg_H,
                    update_H=update_H,
                    verbose=self.verbose,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )
        elif self.solver == "mu":
                W, H, n_iter, D, I = _fit_multiplicative_update(
                    X,
                    W,
                    H,
                    I,
                    D,
                    self._beta_loss,
                    self.max_iter,
                    self.tol,
                    l1_reg_W,
                    l1_reg_H,
                    l2_reg_W,
                    l2_reg_H,
                    update_H,
                    update_W,
                    self.p,
                    self.mu,
                    self.normalize,
                    self.verbose,
                )
        else:
            raise ValueError("Invalid solver parameter '%s'." % self.solver)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                    "Maximum number of iterations %d reached. Increase "
                    "it to improve convergence." % self.max_iter,
                    ConvergenceWarning,
                )

        return W, H, n_iter, D, I

    def transform(self, X, y):
        """Transform the data X according to the fitted NMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        # X = self._validate_data(
        #     X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
        # )

        with config_context(assume_finite=True):
            I = y #init_I(self.component_per_class, X.shape[1], y)
            # I = np.zeros((y.shape[0],y.shape[1]))
            W, H, iter, *_ = self._fit_transform(X, I, W=self.W , update_H=False, update_W=True)



        return H

    # def fit_transform(self, X, y=None, W=None, H=None):
    #     # Profile the fit process
    #     cProfile.runctx('super().fit_transform(X, y, W, H)', globals(), locals(), 'profile_results')
    #     p = pstats.Stats('profile_results')
    #     p.sort_stats('cumulative').print_stats(10)
    #
    #     return super().fit_transform(X, y, W, H)

if __name__ == "__main__":

    import numpy as np
    from sklearn import datasets
    #
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    print(X.shape)
    print(Y)

    model = RSNMF(n_components=3,component_per_class=1, p=2, mu=.001, normalize=True, max_iter=1000)
    W = model.fit_transform(X.T, y=Y)
    H = model.components_








