import logging

import cvxpy
import numpy as np
from asgl import ASGL
from sklearn.base import MultiOutputMixin
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNet
from sklearn.linear_model._base import LinearModel
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted


class AdaptiveElasticNet(ASGL, ElasticNet, MultiOutputMixin, RegressorMixin):
    """
    Objective function is

        (1 / 2 n_samples) sum_i ||y_i - y_pred_i||^2
            + alpha * l1ratio * sum_j |coef_j|
            + alpha * (1 - l1ratio) * sum_j w_j ||coef_j||^2

        w_j = |b_j| ** (-gamma)
        b_j = coefs obtained by fitting ordinary elastic net

        i: sample
        j: feature
        |X|: abs
        ||X||: square norm

    Parameters
    ----------
    - alpha
    - l1_ratio
    - fit_intercept = True
    - gamma
        TODO sensible default value

    Examples
    --------
    >>> np.random.seed(42)

    >>> X = np.random.randn(100, 3)
    >>> y = X[:, 0] + 2 * X[:, 1] + 3

    >>> model = AdaptiveElasticNet().fit(X, y)
    >>> model.coef_
    array([0.        , 0.89455341, 0.        ])
    >>> model.intercept_
    2.864154065031719

    >>> X = np.random.randn(5, 3)
    >>> X[:, 0] + 2 * X[:, 1] + 3
    array([3.56237162, 3.4575266 , 0.43723578, 5.59999799, 3.71857286])
    >>> model.predict(X)
    array([4.10719537, 2.31378436, 1.49776611, 3.54802068, 3.57138261])
    """

    def __init__(self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True, gamma=1.0,eps=1e-5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.gamma = gamma
        self.eps = eps

        # TODO random_state

        self.model = "lm"
        self.solver = "default"
        self.tol = 1e-5
        assert self.fit_intercept

    def fit(self, X, y):
        self.coef_, self.intercept_ = self._ae(X, y)
        return self

    def predict(self, X):
        return super(ElasticNet, self).predict(X)

    def _ae(self, X, y) -> (np.array, float):
        """
        Adaptive elastic-net counterpart of ASGL.asgl

        Returns
        -------
        (coef, intercept)
            - coef : np.array, shape(n_features,)
            - intercept : float
        """
        check_X_y(X, y)

        n_samples, n_features = X.shape
        group_index = np.ones(n_features)
        _, beta_variables = self._num_beta_var_from_group_index(group_index)

        model_prediction = 0.0

        if self.fit_intercept:
            beta_variables = [cvxpy.Variable(1)] + beta_variables
            _1 = cvxpy.Constant(np.ones((n_samples, 1)))
            model_prediction += _1 @ beta_variables[0]

        # --- define objective function ---
        #   l1 weights w_i are identified with coefs in usual elastic net
        #   l2 weights nu_i are fixed to unity in adaptive elastic net

        # /2 * n_samples to make it consistent with sklearn (asgl uses /n_samples)
        model_prediction += X @ beta_variables[1]
        error = cvxpy.sum_squares(y - model_prediction) / (2 * n_samples)

        # XXX: we, paper by Zou Zhang and sklearn use norm squared for l2_penalty whereas asgl uses norm itself
        l1_coefs = self.alpha * self.l1_ratio * self._weights_from_elasticnet(X, y)
        l2_coefs = self.alpha * (1 - self.l1_ratio) * 1.0
        l1_penalty = cvxpy.Constant(l1_coefs) @ cvxpy.abs(beta_variables[1])
        l2_penalty = cvxpy.Constant(l2_coefs) * cvxpy.sum_squares(beta_variables[1])

        # --- optimization ---
        problem = cvxpy.Problem(cvxpy.Minimize(error + l1_penalty + l2_penalty))
        try:
            if self.solver == "default":
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            logging.warning(
                "Default solver failed. Using alternative options. Check solver and solver_stats for more "
                "details"
            )
            solver = ["ECOS", "OSQP", "SCS"]
            for elt in solver:
                solver_dict = self._cvxpy_solver_options(solver=elt)
                try:
                    problem.solve(**solver_dict)
                    if "optimal" in problem.status:
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
        self.solver_stats = problem.solver_stats
        if problem.status in ["infeasible", "unbounded"]:
            logging.warning("Optimization problem status failure")
        beta_sol = np.concatenate([b.value for b in beta_variables], axis=0)
        beta_sol[np.abs(beta_sol) < self.tol] = 0

        intercept, coef = beta_sol[0], beta_sol[1:]

        return (coef, intercept)

    def _weights_from_elasticnet(self, X, y) -> np.array:
        """
        Determine weighs by fitting ElasticNet

        wj of (2.1) in Zou-Zhang 2009

        Returns
        -------
        weights : np.array, shape (n_features,)
        """
        abscoef = np.maximum(np.abs(ElasticNet().fit(X, y).coef_), self.eps)
        weights = 1 / (abscoef ** self.gamma)

        return weights
