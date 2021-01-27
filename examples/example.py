import sys

from sklearn.datasets import make_regression

sys.path.append("..")
from aenet import AdaptiveElasticNet
from aenet import AdaptiveElasticNetCV


if __name__=='__main__':
    X, y = make_regression(n_features=2, random_state=0)

    model = AdaptiveElasticNet()
    model.fit(X, y)
    print(model)
    print(model.coef_)
    print(model.intercept_)
    print(model.predict([[0, 0]]))


    model = AdaptiveElasticNetCV()
    model.fit(X, y)
    print(model)
    print(model.alpha_)
    print(model.coef_)
    print(model.intercept_)
    print(model.predict([[0, 0]]))
