import sys

from sklearn.datasets import make_regression

sys.path.append("..")
from aenet import AdaptiveElasticNet
from aenet import AdaptiveElasticNetCV


if __name__=='__main__':
    X, y = make_regression(n_features=2, random_state=0)

    model = AdaptiveElasticNet()
    print(model.fit(X, y))
    # AdaptiveElasticNet(solver='default', tol=1e-05)
    print(model.coef_)
    # [14.2... 48.9...]
    print(model.intercept_)
    # 2.09...
    print(model.predict([[0, 0]]))
    # [2.09...]

    model = AdaptiveElasticNetCV()
    print(model.fit(X, y))
    # AdaptiveElasticNetCV()
    print(model.alpha_)
    # 0.199...
    print(model.coef_)
    # [24.1... 80.6...]
    print(model.intercept_)
    # 0.706...
    print(model.predict([[0, 0]]))
    # [0.706...]
