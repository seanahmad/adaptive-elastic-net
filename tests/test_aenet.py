import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_raises
from sklearn.utils._testing import assert_raises_regex
from sklearn.utils._testing import assert_raise_message
from sklearn.utils._testing import assert_warns
from sklearn.utils._testing import assert_warns_message
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import assert_array_equal

from aenet import AdaptiveElasticNet


class TestAdaptiveElasticNet:
    """
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/tests/test_coordinate_descent.py
    """

    def test_lasso_zero(self):
        # Check that the aenet can handle zero data without crashing
        X = [[0], [0], [0]]
        y = [0, 0, 0]
        m = AdaptiveElasticNet(alpha=0.1).fit(X, y)
        pred = m.predict([[1], [2], [3]])
        assert_array_almost_equal(m.coef_, [0])
        assert_array_almost_equal(pred, [0, 0, 0])
        # assert_almost_equal(m.dual_gap_, 0)

    def test_positive(self):
        X, y = make_regression(n_features=10, random_state=42)

        model = AdaptiveElasticNet(positive=True).fit(X, y)
        assert (model.coef_ >= 0).all()

        X, y = make_regression(n_features=10, random_state=42)
        y = -1e-10 * y
        model = AdaptiveElasticNet(positive=True).fit(X, y)
        assert (model.coef_ >= 0).all()

        X, y = make_regression(n_features=10, random_state=42)
        with pytest.raises(ValueError):
            model = AdaptiveElasticNet(positive=True).fit(X, -y)
            assert (model.coef_ >= 0).all()

# def test_enet_toy():
#     # Test ElasticNet for various parameters of alpha and l1_ratio.
#     # Actually, the parameters alpha = 0 should not be allowed. However,
#     # we test it as a border case.
#     # ElasticNet is tested with and without precomputed Gram matrix

#     X = np.array([[-1.], [0.], [1.]])
#     Y = [-1, 0, 1]       # just a straight line
#     T = [[2.], [3.], [4.]]  # test sample

#     # this should be the same as lasso
#     clf = AdaptiveElasticNet(alpha=1e-8, l1_ratio=1.0)
#     clf.fit(X, Y)
#     pred = clf.predict(T)
#     assert_array_almost_equal(clf.coef_, [1])
#     assert_array_almost_equal(pred, [2, 3, 4])
#     # assert_almost_equal(clf.dual_gap_, 0)

#     # clf = AdaptiveElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=100, precompute=False)
#     clf = AdaptiveElasticNet(alpha=0.5, l1_ratio=0.3, precompute=False)
#     clf.fit(X, Y)
#     pred = clf.predict(T)
#     assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
#     assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
#     # assert_almost_equal(clf.dual_gap_, 0)

#     # clf.set_params(max_iter=100, precompute=True)
#     clf.set_params(precompute=True)
#     clf.fit(X, Y)  # with Gram
#     pred = clf.predict(T)
#     assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
#     assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
#     # assert_almost_equal(clf.dual_gap_, 0)

#     # clf.set_params(max_iter=100, precompute=np.dot(X.T, X))
#     clf.set_params(precompute=np.dot(X.T, X))
#     clf.fit(X, Y)  # with Gram
#     pred = clf.predict(T)
#     assert_array_almost_equal(clf.coef_, [0.50819], decimal=3)
#     assert_array_almost_equal(pred, [1.0163, 1.5245, 2.0327], decimal=3)
#     # assert_almost_equal(clf.dual_gap_, 0)

#     clf = AdaptiveElasticNet(alpha=0.5, l1_ratio=0.5)
#     clf.fit(X, Y)
#     pred = clf.predict(T)
#     assert_array_almost_equal(clf.coef_, [0.45454], 3)
#     assert_array_almost_equal(pred, [0.9090, 1.3636, 1.8181], 3)
#     # assert_almost_equal(clf.dual_gap_, 0)
