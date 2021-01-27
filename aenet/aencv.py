import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model._coordinate_descent import LinearModelCV

from .aen import AdaptiveElasticNet


class AdaptiveElasticNetCV(RegressorMixin, LinearModelCV):
    """
    AdaptiveElasticNet with CV

    Parameters
    ----------

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
        normalize=False,
        precompute="auto",
        cv=None,
        # copy_X=True,
        # selection="cyclic",
        eps=1e-3,
    ):
        # --- initialize ---
        # l1 float ok
        # cv int ok
        # eps ???
        # positive ??? important
        # max_iter cvx ni aru
        # precompute optional
        # jobs optional
        # positive  see elasticnet
        # ElasticNetCV(l1_ratio=l1, cv=self.cv, positive=True, eps=0.003, max_iter=100000, precompute=True, n_jobs=3)
        super().__init__(
            n_alphas=n_alphas,
            alphas=alphas,
            fit_intercept=fit_intercept,
            normalize=normalize,
            precompute=precompute,
            cv=cv,
            # copy_X=copy_X,
            # selection=selection,
        )

        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.eps = eps

    def _get_estimator(self):
        # TODO check that these values are reflected in cv
        return AdaptiveElasticNet(
            l1_ratio=self.l1_ratio, gamma=self.gamma, eps=self.eps
        )

    def _is_multitask(self):
        return False
