# Adaptive Elastic-Net

## Install

```sh
pip install aenet
```

## How to use

```py
from aenet import AdaptiveElasticNet
from aenet import AdaptiveElasticNetCV

X, y = ...

model = AdaptiveElasticNet().fit(X, y)
model.predict(X)
model.score(X)

model = AdaptiveElasticNetCV().fit(X, y)
model.alpha_
model.predict(X)
model.score(X)
```
