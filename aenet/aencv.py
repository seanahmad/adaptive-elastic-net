from sklearn.base import RegressorMixin
from sklearn.linear_model._coordinate_descent import LinearModelCV

from .aen import AdaptiveElasticNet


class AdaptiveElasticNetCV(RegressorMixin, LinearModelCV):
    """
    AdaptiveElasticNet with CV

    Parameters
    ----------
    - l1_ratio : float, default=0.5
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
    - n_alphas :
    - alphas :
    - gamma : float, default 1.0
        To guarantee the oracle property, following inequality should be satisfied:
            gamma > 2 * nu / (1 - nu)
            nu = lim(n_samples -> inf) [log(n_features) / log(n_samples)]
        default is 1 because this value is natural in the sense that
        l1_penalty / l2_penalty is not (directly) dependent on scale of features
    - fit_intercept : bool, default True
        Whether to calculate the intercept for this model.
        For now False is not allowed
    - cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
    - eps : float, default=1e-3
        Length of the path.
        eps=1e-3 means that alpha_min / alpha_max = 1e-3.
    - positive : bool, default=False
        When set to True, forces the coefficients to be positive.
    - positive_tol : float, optional
        Numerical optimization (cvxpy) may return slightly negative coefs.
        (See cvxpy issue/#1201)
        If coef > -positive_tol, ignore this and forcively set negative coef to zero.
        Otherwise, raise ValueError.
        If `positive_tol=None` always ignore (default)

    TODO
    ----
    cv wrt gamma?

    Notes
    -----
    (simaki)
        accoding to https://projecteuclid.org/download/pdfview_1/euclid.aos/1245332831
        condition gamma > 2 nu / 1 - nu is necessary to guarantee oracle property
        nu = log(n_features) / log(n_samples), which is in range (0, 1)

        hmm:
        Also note that, in the finite dimension setting, ν = 0; thus, any positive gamma
        can be used, which agrees with the results in Zou (2006).

    Examples
    --------
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> model = AdaptiveElasticNetCV(cv=5)
    >>> model.fit(X, y)
    AdaptiveElasticNetCV(cv=5)
    >>> print(model.alpha_)
    0.199...
    >>> print(model.intercept_)
    0.706...
    >>> print(model.predict([[0, 0]]))
    [0.706...]

    >>> X, y = make_regression(n_features=10, random_state=0)
    >>> model = AdaptiveElasticNetCV(positive=True).fit(X, -y)
    >>> model.coef_
    array([1.16980429e-04, 2.14503535e-05, 0.00000000e+00, 4.45525264e-05,
           3.00411576e-04, 1.26646882e-04, 0.00000000e+00, 1.42388065e-04,
           2.05464198e-03, 0.00000000e+00])
    """

    path = AdaptiveElasticNet.aenet_path

    def __init__(
        self,
        *,
        l1_ratio=0.5,
        n_alphas=100,
        alphas=None,
        gamma=1.0,
        fit_intercept=True,
        # precompute="auto",
        cv=None,
        # copy_X=True,
        # selection="cyclic",
        eps=1e-3,
        positive=False,
        positive_tol=None,
        normalize=False,
        precompute="auto",
    ):
        super().__init__(
            n_alphas=n_alphas,
            alphas=alphas,
            fit_intercept=fit_intercept,
            normalize=normalize,
            precompute=precompute,
            # precompute=precompute,
            cv=cv,
            # copy_X=copy_X,
            # selection=selection,
            eps=eps,
        )

        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.eps = eps
        self.positive = positive
        self.positive_tol = positive_tol

    def _get_estimator(self):
        return AdaptiveElasticNet(
            l1_ratio=self.l1_ratio,
            gamma=self.gamma,
            positive=self.positive,
            positive_tol=self.positive_tol,
        )

    def _is_multitask(self):
        return False
