import pytest
from sklearn.datasets import make_regression

from aenet import AdaptiveElasticNet


class TestAdaptiveElasticNet:
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
