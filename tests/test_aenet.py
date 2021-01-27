"""
We use scikit-learn, which is under:

BSD 3-Clause License

Copyright (c) 2007-2020 The scikit-learn developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_raise_message
from sklearn.utils._testing import assert_raises
from sklearn.utils._testing import assert_raises_regex
from sklearn.utils._testing import assert_warns
from sklearn.utils._testing import assert_warns_message
from sklearn.utils._testing import ignore_warnings
from sklearn.linear_model.tests.test_coordinate_descent import build_dataset

from aenet import AdaptiveElasticNet
from aenet import AdaptiveElasticNetCV


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


@pytest.mark.parametrize("l1_ratio", (-1, 2, None, 10, "something_wrong"))
def test_l1_ratio_param_invalid(l1_ratio):
    # Check that correct error is raised when l1_ratio in ElasticNet
    # is outside the correct range
    X = np.array([[-1.0], [0.0], [1.0]])
    Y = [-1, 0, 1]  # just a straight line

    msg = "l1_ratio must be between 0 and 1; got l1_ratio="
    clf = AdaptiveElasticNet(alpha=0.1, l1_ratio=l1_ratio)
    with pytest.raises(ValueError, match=msg):
        clf.fit(X, Y)


def test_enet_toy():
    """
    https://github.com/scikit-learn/scikit-learn/blob/0aee596bb32136df8c68371d696770251c7d14a0/sklearn/linear_model/tests/test_coordinate_descent.py#L161
    """
    # Test ElasticNet for various parameters of alpha and l1_ratio.
    # Actually, the parameters alpha = 0 should not be allowed. However,
    # we test it as a border case.
    # ElasticNet is tested with and without precomputed Gram matrix

    X = np.array([[-1.0], [0.0], [1.0]])
    Y = [-1, 0, 1]  # just a straight line
    T = [[2.0], [3.0], [4.0]]  # test sample

    # this should be the same as lasso
    clf = AdaptiveElasticNet(alpha=1e-8, l1_ratio=1.0)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(pred, [2, 3, 4])
    # assert_almost_equal(clf.dual_gap_, 0)

    # clf = AdaptiveElasticNet(alpha=0.5, l1_ratio=0.3, max_iter=100, precompute=False)
    clf = AdaptiveElasticNet(alpha=0.5, l1_ratio=0.3, precompute=False)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0], decimal=3)
    assert_array_almost_equal(pred, [0.0, 0.0, 0.0], decimal=3)
    # assert_almost_equal(clf.dual_gap_, 0)

    # clf.set_params(max_iter=100, precompute=True)
    clf.set_params(precompute=True)
    clf.fit(X, Y)  # with Gram
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0], decimal=3)
    assert_array_almost_equal(pred, [0.0, 0.0, 0.0], decimal=3)
    # assert_almost_equal(clf.dual_gap_, 0)

    # clf.set_params(max_iter=100, precompute=np.dot(X.T, X))
    clf.set_params(precompute=np.dot(X.T, X))
    clf.fit(X, Y)  # with Gram
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0], decimal=3)
    assert_array_almost_equal(pred, [0.0, 0.0, 0.0], decimal=3)
    # assert_almost_equal(clf.dual_gap_, 0)

    clf = AdaptiveElasticNet(alpha=0.5, l1_ratio=0.5)
    clf.fit(X, Y)
    pred = clf.predict(T)
    assert_array_almost_equal(clf.coef_, [0.0], 3)
    assert_array_almost_equal(pred, [0.0, 0.0, 0.0], 3)
    # assert_almost_equal(clf.dual_gap_, 0)


# def test_enet_path():
#     # We use a large number of samples and of informative features so that
#     # the l1_ratio selected is more toward ridge than lasso
#     X, y, X_test, y_test = build_dataset(
#         n_samples=200, n_features=100, n_informative_features=100
#     )
#     max_iter = 150

#     # Here we have a small number of iterations, and thus the
#     # ElasticNet might not converge. This is to speed up tests
#     clf = ElasticNetCV(
#         alphas=[0.01, 0.05, 0.1], eps=2e-3, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter
#     )
#     ignore_warnings(clf.fit)(X, y)
#     # Well-conditioned settings, we should have selected our
#     # smallest penalty
#     assert_almost_equal(clf.alpha_, min(clf.alphas_))
#     # Non-sparse ground truth: we should have selected an elastic-net
#     # that is closer to ridge than to lasso
#     assert clf.l1_ratio_ == min(clf.l1_ratio)

#     clf = ElasticNetCV(
#         alphas=[0.01, 0.05, 0.1],
#         eps=2e-3,
#         l1_ratio=[0.5, 0.7],
#         cv=3,
#         max_iter=max_iter,
#         precompute=True,
#     )
#     ignore_warnings(clf.fit)(X, y)

#     # Well-conditioned settings, we should have selected our
#     # smallest penalty
#     assert_almost_equal(clf.alpha_, min(clf.alphas_))
#     # Non-sparse ground truth: we should have selected an elastic-net
#     # that is closer to ridge than to lasso
#     assert clf.l1_ratio_ == min(clf.l1_ratio)

#     # We are in well-conditioned settings with low noise: we should
#     # have a good test-set performance
#     assert clf.score(X_test, y_test) > 0.99

#     # Multi-output/target case
#     X, y, X_test, y_test = build_dataset(n_features=10, n_targets=3)
#     clf = MultiTaskElasticNetCV(
#         n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7], cv=3, max_iter=max_iter
#     )
#     ignore_warnings(clf.fit)(X, y)
#     # We are in well-conditioned settings with low noise: we should
#     # have a good test-set performance
#     assert clf.score(X_test, y_test) > 0.99
#     assert clf.coef_.shape == (3, 10)

#     # Mono-output should have same cross-validated alpha_ and l1_ratio_
#     # in both cases.
#     X, y, _, _ = build_dataset(n_features=10)
#     clf1 = ElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
#     clf1.fit(X, y)
#     clf2 = MultiTaskElasticNetCV(n_alphas=5, eps=2e-3, l1_ratio=[0.5, 0.7])
#     clf2.fit(X, y[:, np.newaxis])
#     assert_almost_equal(clf1.l1_ratio_, clf2.l1_ratio_)
#     assert_almost_equal(clf1.alpha_, clf2.alpha_)


def test_path_parameters():
    X, y, _, _ = build_dataset()
    max_iter = 100

    # FIXME when max_iter and tol are implemented
    # clf = AdaptiveElasticNetCV(n_alphas=50, eps=1e-3, max_iter=max_iter, l1_ratio=0.5, tol=1e-3)
    clf = AdaptiveElasticNetCV(n_alphas=50, eps=1e-3, l1_ratio=0.5)
    clf.fit(X, y)  # new params
    assert_almost_equal(0.5, clf.l1_ratio)
    assert 50 == clf.n_alphas
    assert 50 == len(clf.alphas_)


# def test_warm_start():
#     X, y, _, _ = build_dataset()
#     clf = ElasticNet(alpha=0.1, max_iter=5, warm_start=True)
#     ignore_warnings(clf.fit)(X, y)
#     ignore_warnings(clf.fit)(X, y)  # do a second round with 5 iterations

#     clf2 = ElasticNet(alpha=0.1, max_iter=10)
#     ignore_warnings(clf2.fit)(X, y)
#     assert_array_almost_equal(clf2.coef_, clf.coef_)


# def test_enet_positive_constraint():
#     X = [[-1], [0], [1]]
#     y = [1, 0, -1]  # just a straight line with negative slope

#     enet = AdaptiveElasticNet(alpha=0.1, max_iter=1000, positive=True)
#     enet.fit(X, y)
#     assert min(enet.coef_) >= 0


# def test_enet_cv_positive_constraint():
#     X, y, X_test, y_test = build_dataset()
#     max_iter = 500

#     # Ensure the unconstrained fit has a negative coefficient
#     enetcv_unconstrained = ElasticNetCV(
#         n_alphas=3, eps=1e-1, max_iter=max_iter, cv=2, n_jobs=1
#     )
#     enetcv_unconstrained.fit(X, y)
#     assert min(enetcv_unconstrained.coef_) < 0

#     # On same data, constrained fit has non-negative coefficients
#     enetcv_constrained = ElasticNetCV(
#         n_alphas=3, eps=1e-1, max_iter=max_iter, cv=2, positive=True, n_jobs=1
#     )
#     enetcv_constrained.fit(X, y)
#     assert min(enetcv_constrained.coef_) >= 0


# def test_uniform_targets():
#     enet = ElasticNetCV(n_alphas=3)
#     m_enet = MultiTaskElasticNetCV(n_alphas=3)
#     lasso = LassoCV(n_alphas=3)
#     m_lasso = MultiTaskLassoCV(n_alphas=3)

#     models_single_task = (enet, lasso)
#     models_multi_task = (m_enet, m_lasso)

#     rng = np.random.RandomState(0)

#     X_train = rng.random_sample(size=(10, 3))
#     X_test = rng.random_sample(size=(10, 3))

#     y1 = np.empty(10)
#     y2 = np.empty((10, 2))

#     for model in models_single_task:
#         for y_values in (0, 5):
#             y1.fill(y_values)
#             assert_array_equal(model.fit(X_train, y1).predict(X_test), y1)
#             assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)

#     for model in models_multi_task:
#         for y_values in (0, 5):
#             y2[:, 0].fill(y_values)
#             y2[:, 1].fill(2 * y_values)
#             assert_array_equal(model.fit(X_train, y2).predict(X_test), y2)
#             assert_array_equal(model.alphas_, [np.finfo(float).resolution] * 3)


# def test_multi_task_lasso_and_enet():
#     X, y, X_test, y_test = build_dataset()
#     Y = np.c_[y, y]
#     # Y_test = np.c_[y_test, y_test]
#     clf = MultiTaskLasso(alpha=1, tol=1e-8).fit(X, Y)
#     assert 0 < clf.dual_gap_ < 1e-5
#     assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

#     clf = MultiTaskElasticNet(alpha=1, tol=1e-8).fit(X, Y)
#     assert 0 < clf.dual_gap_ < 1e-5
#     assert_array_almost_equal(clf.coef_[0], clf.coef_[1])

#     clf = MultiTaskElasticNet(alpha=1.0, tol=1e-8, max_iter=1)
#     assert_warns_message(ConvergenceWarning, "did not converge", clf.fit, X, Y)


# def test_lasso_readonly_data():
#     X = np.array([[-1], [0], [1]])
#     Y = np.array([-1, 0, 1])  # just a straight line
#     T = np.array([[2], [3], [4]])  # test sample
#     with TempMemmap((X, Y)) as (X, Y):
#         clf = Lasso(alpha=0.5)
#         clf.fit(X, Y)
#         pred = clf.predict(T)
#         assert_array_almost_equal(clf.coef_, [0.25])
#         assert_array_almost_equal(pred, [0.5, 0.75, 1.0])
#         assert_almost_equal(clf.dual_gap_, 0)


# def test_multi_task_lasso_readonly_data():
#     X, y, X_test, y_test = build_dataset()
#     Y = np.c_[y, y]
#     with TempMemmap((X, Y)) as (X, Y):
#         Y = np.c_[y, y]
#         clf = MultiTaskLasso(alpha=1, tol=1e-8).fit(X, Y)
#         assert 0 < clf.dual_gap_ < 1e-5
#         assert_array_almost_equal(clf.coef_[0], clf.coef_[1])


# def test_elasticnet_precompute_incorrect_gram():
#     # check that passing an invalid precomputed Gram matrix will raise an
#     # error.
#     X, y, _, _ = build_dataset()

#     rng = np.random.RandomState(0)

#     X_centered = X - np.average(X, axis=0)
#     garbage = rng.standard_normal(X.shape)
#     precompute = np.dot(garbage.T, garbage)

#     clf = ElasticNet(alpha=0.01, precompute=precompute)
#     msg = "Gram matrix.*did not pass validation.*"
#     with pytest.raises(ValueError, match=msg):
#         clf.fit(X_centered, y)


# def test_elasticnet_precompute_gram_weighted_samples():
#     # check the equivalence between passing a precomputed Gram matrix and
#     # internal computation using sample weights.
#     X, y, _, _ = build_dataset()

#     rng = np.random.RandomState(0)
#     sample_weight = rng.lognormal(size=y.shape)

#     w_norm = sample_weight * (y.shape / np.sum(sample_weight))
#     X_c = X - np.average(X, axis=0, weights=w_norm)
#     X_r = X_c * np.sqrt(w_norm)[:, np.newaxis]
#     gram = np.dot(X_r.T, X_r)

#     clf1 = ElasticNet(alpha=0.01, precompute=gram)
#     clf1.fit(X_c, y, sample_weight=sample_weight)

#     clf2 = ElasticNet(alpha=0.01, precompute=False)
#     clf2.fit(X, y, sample_weight=sample_weight)

#     assert_allclose(clf1.coef_, clf2.coef_)


# def test_warm_start_convergence():
#     X, y, _, _ = build_dataset()
#     model = ElasticNet(alpha=1e-3, tol=1e-3).fit(X, y)
#     n_iter_reference = model.n_iter_

#     # This dataset is not trivial enough for the model to converge in one pass.
#     assert n_iter_reference > 2

#     # Check that n_iter_ is invariant to multiple calls to fit
#     # when warm_start=False, all else being equal.
#     model.fit(X, y)
#     n_iter_cold_start = model.n_iter_
#     assert n_iter_cold_start == n_iter_reference

#     # Fit the same model again, using a warm start: the optimizer just performs
#     # a single pass before checking that it has already converged
#     model.set_params(warm_start=True)
#     model.fit(X, y)
#     n_iter_warm_start = model.n_iter_
#     assert n_iter_warm_start == 1


# def test_warm_start_convergence_with_regularizer_decrement():
#     X, y = load_diabetes(return_X_y=True)

#     # Train a model to converge on a lightly regularized problem
#     final_alpha = 1e-5
#     low_reg_model = ElasticNet(alpha=final_alpha).fit(X, y)

#     # Fitting a new model on a more regularized version of the same problem.
#     # Fitting with high regularization is easier it should converge faster
#     # in general.
#     high_reg_model = ElasticNet(alpha=final_alpha * 10).fit(X, y)
#     assert low_reg_model.n_iter_ > high_reg_model.n_iter_

#     # Fit the solution to the original, less regularized version of the
#     # problem but from the solution of the highly regularized variant of
#     # the problem as a better starting point. This should also converge
#     # faster than the original model that starts from zero.
#     warm_low_reg_model = deepcopy(high_reg_model)
#     warm_low_reg_model.set_params(warm_start=True, alpha=final_alpha)
#     warm_low_reg_model.fit(X, y)
#     assert low_reg_model.n_iter_ > warm_low_reg_model.n_iter_


# def test_random_descent():
#     # Test that both random and cyclic selection give the same results.
#     # Ensure that the test models fully converge and check a wide
#     # range of conditions.

#     # This uses the coordinate descent algo using the gram trick.
#     X, y, _, _ = build_dataset(n_samples=50, n_features=20)
#     clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
#     clf_cyclic.fit(X, y)
#     clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
#     clf_random.fit(X, y)
#     assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
#     assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

#     # This uses the descent algo without the gram trick
#     clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
#     clf_cyclic.fit(X.T, y[:20])
#     clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
#     clf_random.fit(X.T, y[:20])
#     assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
#     assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

#     # Sparse Case
#     clf_cyclic = ElasticNet(selection="cyclic", tol=1e-8)
#     clf_cyclic.fit(sparse.csr_matrix(X), y)
#     clf_random = ElasticNet(selection="random", tol=1e-8, random_state=42)
#     clf_random.fit(sparse.csr_matrix(X), y)
#     assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
#     assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

#     # Multioutput case.
#     new_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
#     clf_cyclic = MultiTaskElasticNet(selection="cyclic", tol=1e-8)
#     clf_cyclic.fit(X, new_y)
#     clf_random = MultiTaskElasticNet(selection="random", tol=1e-8, random_state=42)
#     clf_random.fit(X, new_y)
#     assert_array_almost_equal(clf_cyclic.coef_, clf_random.coef_)
#     assert_almost_equal(clf_cyclic.intercept_, clf_random.intercept_)

#     # Raise error when selection is not in cyclic or random.
#     clf_random = ElasticNet(selection="invalid")
#     assert_raises(ValueError, clf_random.fit, X, y)


# def test_enet_path_positive():
#     # Test positive parameter

#     X, Y, _, _ = build_dataset(n_samples=50, n_features=50, n_targets=2)

#     # For mono output
#     # Test that the coefs returned by positive=True in enet_path are positive
#     for path in [enet_path, lasso_path]:
#         pos_path_coef = path(X, Y[:, 0], positive=True)[1]
#         assert np.all(pos_path_coef >= 0)

#     # For multi output, positive parameter is not allowed
#     # Test that an error is raised
#     for path in [enet_path, lasso_path]:
#         assert_raises(ValueError, path, X, Y, positive=True)


# @pytest.mark.parametrize("check_input", [True, False])
# def test_enet_copy_X_True(check_input):
#     X, y, _, _ = build_dataset()
#     X = X.copy(order="F")

#     original_X = X.copy()
#     enet = ElasticNet(copy_X=True)
#     enet.fit(X, y, check_input=check_input)

#     assert_array_equal(original_X, X)


# def test_enet_copy_X_False_check_input_False():
#     X, y, _, _ = build_dataset()
#     X = X.copy(order="F")

#     original_X = X.copy()
#     enet = ElasticNet(copy_X=False)
#     enet.fit(X, y, check_input=False)

#     # No copying, X is overwritten
#     assert np.any(np.not_equal(original_X, X))


# def test_overrided_gram_matrix():
#     X, y, _, _ = build_dataset(n_samples=20, n_features=10)
#     Gram = X.T.dot(X)
#     clf = ElasticNet(selection="cyclic", tol=1e-8, precompute=Gram)
#     assert_warns_message(
#         UserWarning,
#         "Gram matrix was provided but X was centered"
#         " to fit intercept, "
#         "or X was normalized : recomputing Gram matrix.",
#         clf.fit,
#         X,
#         y,
#     )


# @pytest.mark.parametrize("model", [ElasticNet, Lasso])
# def test_lasso_non_float_y(model):
#     X = [[0, 0], [1, 1], [-1, -1]]
#     y = [0, 1, 2]
#     y_float = [0.0, 1.0, 2.0]

#     clf = model(fit_intercept=False)
#     clf.fit(X, y)
#     clf_float = model(fit_intercept=False)
#     clf_float.fit(X, y_float)
#     assert_array_equal(clf.coef_, clf_float.coef_)


# def test_enet_float_precision():
#     # Generate dataset
#     X, y, X_test, y_test = build_dataset(n_samples=20, n_features=10)
#     # Here we have a small number of iterations, and thus the
#     # ElasticNet might not converge. This is to speed up tests

#     for normalize in [True, False]:
#         for fit_intercept in [True, False]:
#             coef = {}
#             intercept = {}
#             for dtype in [np.float64, np.float32]:
#                 clf = ElasticNet(
#                     alpha=0.5,
#                     max_iter=100,
#                     precompute=False,
#                     fit_intercept=fit_intercept,
#                     normalize=normalize,
#                 )

#                 X = dtype(X)
#                 y = dtype(y)
#                 ignore_warnings(clf.fit)(X, y)

#                 coef[("simple", dtype)] = clf.coef_
#                 intercept[("simple", dtype)] = clf.intercept_

#                 assert clf.coef_.dtype == dtype

#                 # test precompute Gram array
#                 Gram = X.T.dot(X)
#                 clf_precompute = ElasticNet(
#                     alpha=0.5,
#                     max_iter=100,
#                     precompute=Gram,
#                     fit_intercept=fit_intercept,
#                     normalize=normalize,
#                 )
#                 ignore_warnings(clf_precompute.fit)(X, y)
#                 assert_array_almost_equal(clf.coef_, clf_precompute.coef_)
#                 assert_array_almost_equal(clf.intercept_, clf_precompute.intercept_)

#                 # test multi task enet
#                 multi_y = np.hstack((y[:, np.newaxis], y[:, np.newaxis]))
#                 clf_multioutput = MultiTaskElasticNet(
#                     alpha=0.5,
#                     max_iter=100,
#                     fit_intercept=fit_intercept,
#                     normalize=normalize,
#                 )
#                 clf_multioutput.fit(X, multi_y)
#                 coef[("multi", dtype)] = clf_multioutput.coef_
#                 intercept[("multi", dtype)] = clf_multioutput.intercept_
#                 assert clf.coef_.dtype == dtype

#             for v in ["simple", "multi"]:
#                 assert_array_almost_equal(
#                     coef[(v, np.float32)], coef[(v, np.float64)], decimal=4
#                 )
#                 assert_array_almost_equal(
#                     intercept[(v, np.float32)], intercept[(v, np.float64)], decimal=4
#                 )


# def test_enet_l1_ratio():
#     # Test that an error message is raised if an estimator that
#     # uses _alpha_grid is called with l1_ratio=0
#     msg = (
#         "Automatic alpha grid generation is not supported for l1_ratio=0. "
#         "Please supply a grid by providing your estimator with the "
#         "appropriate `alphas=` argument."
#     )
#     X = np.array([[1, 2, 4, 5, 8], [3, 5, 7, 7, 8]]).T
#     y = np.array([12, 10, 11, 21, 5])

#     assert_raise_message(
#         ValueError, msg, ElasticNetCV(l1_ratio=0, random_state=42).fit, X, y
#     )
#     assert_raise_message(
#         ValueError,
#         msg,
#         MultiTaskElasticNetCV(l1_ratio=0, random_state=42).fit,
#         X,
#         y[:, None],
#     )

#     # Test that l1_ratio=0 is allowed if we supply a grid manually
#     alphas = [0.1, 10]
#     estkwds = {"alphas": alphas, "random_state": 42}
#     est_desired = ElasticNetCV(l1_ratio=0.00001, **estkwds)
#     est = ElasticNetCV(l1_ratio=0, **estkwds)
#     with ignore_warnings():
#         est_desired.fit(X, y)
#         est.fit(X, y)
#     assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)

#     est_desired = MultiTaskElasticNetCV(l1_ratio=0.00001, **estkwds)
#     est = MultiTaskElasticNetCV(l1_ratio=0, **estkwds)
#     with ignore_warnings():
#         est.fit(X, y[:, None])
#         est_desired.fit(X, y[:, None])
#     assert_array_almost_equal(est.coef_, est_desired.coef_, decimal=5)


# def test_convergence_warnings():
#     random_state = np.random.RandomState(0)
#     X = random_state.standard_normal((1000, 500))
#     y = random_state.standard_normal((1000, 3))

#     # check that the model fails to converge (a negative dual gap cannot occur)
#     with pytest.warns(ConvergenceWarning):
#         MultiTaskElasticNet(max_iter=1, tol=-1).fit(X, y)

#     # check that the model converges w/o warnings
#     with pytest.warns(None) as record:
#         MultiTaskElasticNet(max_iter=1000).fit(X, y)

#     assert not record.list


# def test_sparse_input_convergence_warning():
#     X, y, _, _ = build_dataset(n_samples=1000, n_features=500)

#     with pytest.warns(ConvergenceWarning):
#         ElasticNet(max_iter=1, tol=0).fit(sparse.csr_matrix(X, dtype=np.float32), y)

#     # check that the model converges w/o warnings
#     with pytest.warns(None) as record:
#         Lasso(max_iter=1000).fit(sparse.csr_matrix(X, dtype=np.float32), y)

#     assert not record.list


# @pytest.mark.parametrize("fit_intercept", [True, False])
# @pytest.mark.parametrize("alpha", [0.01])
# @pytest.mark.parametrize("normalize", [False, True])
# @pytest.mark.parametrize("precompute", [False, True])
# def test_enet_sample_weight_consistency(fit_intercept, alpha, normalize, precompute):
#     """Test that the impact of sample_weight is consistent."""
#     rng = np.random.RandomState(0)
#     n_samples, n_features = 10, 5

#     X = rng.rand(n_samples, n_features)
#     y = rng.rand(n_samples)
#     params = dict(
#         alpha=alpha,
#         fit_intercept=fit_intercept,
#         precompute=precompute,
#         tol=1e-6,
#         l1_ratio=0.5,
#     )

#     reg = ElasticNet(**params).fit(X, y)
#     coef = reg.coef_.copy()
#     if fit_intercept:
#         intercept = reg.intercept_

#     # sample_weight=np.ones(..) should be equivalent to sample_weight=None
#     sample_weight = np.ones_like(y)
#     reg.fit(X, y, sample_weight=sample_weight)
#     assert_allclose(reg.coef_, coef, rtol=1e-6)
#     if fit_intercept:
#         assert_allclose(reg.intercept_, intercept)

#     # sample_weight=None should be equivalent to sample_weight = number
#     sample_weight = 123.0
#     reg.fit(X, y, sample_weight=sample_weight)
#     assert_allclose(reg.coef_, coef, rtol=1e-6)
#     if fit_intercept:
#         assert_allclose(reg.intercept_, intercept)

#     # scaling of sample_weight should have no effect, cf. np.average()
#     sample_weight = 2 * np.ones_like(y)
#     reg.fit(X, y, sample_weight=sample_weight)
#     assert_allclose(reg.coef_, coef, rtol=1e-6)
#     if fit_intercept:
#         assert_allclose(reg.intercept_, intercept)

#     # setting one element of sample_weight to 0 is equivalent to removing
#     # the corresponding sample
#     sample_weight = np.ones_like(y)
#     sample_weight[-1] = 0
#     reg.fit(X, y, sample_weight=sample_weight)
#     coef1 = reg.coef_.copy()
#     if fit_intercept:
#         intercept1 = reg.intercept_
#     reg.fit(X[:-1], y[:-1])
#     assert_allclose(reg.coef_, coef1, rtol=1e-6)
#     if fit_intercept:
#         assert_allclose(reg.intercept_, intercept1)

#     # check that multiplying sample_weight by 2 is equivalent
#     # to repeating corresponding samples twice
#     if sparse.issparse(X):
#         X = X.toarray()

#     X2 = np.concatenate([X, X[: n_samples // 2]], axis=0)
#     y2 = np.concatenate([y, y[: n_samples // 2]])
#     sample_weight_1 = np.ones(len(y))
#     sample_weight_1[: n_samples // 2] = 2

#     reg1 = ElasticNet(**params).fit(X, y, sample_weight=sample_weight_1)

#     reg2 = ElasticNet(**params).fit(X2, y2, sample_weight=None)
#     assert_allclose(reg1.coef_, reg2.coef_)


# def test_enet_sample_weight_sparse():
#     reg = ElasticNet()
#     X = sparse.csc_matrix(np.zeros((3, 2)))
#     y = np.array([-1, 0, 1])
#     sw = np.array([1, 2, 3])
#     with pytest.raises(
#         ValueError, match="Sample weights do not.*support " "sparse matrices"
#     ):
#         reg.fit(X, y, sample_weight=sw, check_input=True)


# @pytest.mark.parametrize("backend", ["loky", "threading"])
# @pytest.mark.parametrize(
#     "estimator", [ElasticNetCV, MultiTaskElasticNetCV, LassoCV, MultiTaskLassoCV]
# )
# def test_linear_models_cv_fit_for_all_backends(backend, estimator):
#     # LinearModelsCV.fit performs inplace operations on input data which is
#     # memmapped when using loky backend, causing an error due to unexpected
#     # behavior of fancy indexing of read-only memmaps (cf. numpy#14132).

#     if parse_version(joblib.__version__) < parse_version("0.12") and backend == "loky":
#         pytest.skip("loky backend does not exist in joblib <0.12")

#     # Create a problem sufficiently large to cause memmapping (1MB).
#     n_targets = 1 + (estimator in (MultiTaskElasticNetCV, MultiTaskLassoCV))
#     X, y = make_regression(20000, 10, n_targets=n_targets)

#     with joblib.parallel_backend(backend=backend):
#         estimator(n_jobs=2, cv=3).fit(X, y)


# @pytest.mark.parametrize("check_input", [True, False])
# def test_enet_sample_weight_does_not_overwrite_sample_weight(check_input):
#     """Check that ElasticNet does not overwrite sample_weights."""

#     rng = np.random.RandomState(0)
#     n_samples, n_features = 10, 5

#     X = rng.rand(n_samples, n_features)
#     y = rng.rand(n_samples)

#     sample_weight_1_25 = 1.25 * np.ones_like(y)
#     sample_weight = sample_weight_1_25.copy()

#     reg = ElasticNet()
#     reg.fit(X, y, sample_weight=sample_weight, check_input=check_input)

#     assert_array_equal(sample_weight, sample_weight_1_25)
