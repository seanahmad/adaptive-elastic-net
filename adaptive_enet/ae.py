import logging

import cvxpy
import numpy as np
from asgl import ASGL
from sklearn.linear_model import ElasticNet


class AdaptiveElasticNet(ASGL):
    """
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
    """

    def __init__(self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept

        self.gamma = 1.0

        # Alias to make it compatible with asgl API
        self.intercept = self.fit_intercept

        self.model = "lm"
        self.solver = "default"
        self.tol = 1e-5

        # TODO random_state

    def fit(self, X, y):
        ae = self._ae(X, y)

        self.coef_ = ae[1:]
        self.intercept_ = ae[0]

        return self

    def _ae(self, X, y) -> np.array:
        """
        Adaptive elastic-net counterpart of ASGL.asgl

        Returns
        -------
        beta : np.array, shape (n_features + n_intercept,)
        """
        n_samples, n_features = X.shape
        group_index = np.ones(n_features)
        _, beta_variables = self._num_beta_var_from_group_index(group_index)

        model_prediction = 0.0

        if self.fit_intercept:
            beta_variables = [cvxpy.Variable(1)] + beta_variables
            model_prediction += (
                cvxpy.Constant(np.ones((n_samples, 1))) @ beta_variables[0]
            )

        # --- define objective function ---
        #   l1 weights w_i are identified with coefs in usual elastic net
        #   l2 weights nu_i are fixed to unity in adaptive elastic net
        model_prediction += X @ beta_variables[1]
        error = cvxpy.sum_squares(y - model_prediction)
        # /2 * n_samples to make it consistent with sklearn (asgl uses /n_samples)
        objective_function = error / (2 * n_samples)

        # XXX: we, paper by Zou Zhang and sklearn use norm squared whereas asgl uses norm itself
        l1_coefs = self.alpha * self.l1_ratio * self._weights_from_elasticnet(X, y)
        l2_coefs = self.alpha * (1 - self.l1_ratio) * 1.0
        l1_penalty = cvxpy.Constant(l1_coefs) @ cvxpy.abs(beta_variables[1])
        l2_penalty = cvxpy.Constant(l2_coefs) * cvxpy.sum_squares(beta_variables[1])

        assert self.model == "lm", "model should be lm"

        objective = cvxpy.Minimize(objective_function + l1_penalty + l2_penalty)
        problem = cvxpy.Problem(objective)

        # --- optimization ---
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

        return beta_sol

    def _weights_from_elasticnet(self, X, y) -> np.array:
        eps = 1e-5
        coef = ElasticNet().fit(X, y).coef_
        coef = np.where(np.abs(coef) > eps, coef, eps)
        weights = coef ** (-self.gamma)
        return weights

    def predict(self, X, y):
        y_pred_list = super().predict(self, X, y)
