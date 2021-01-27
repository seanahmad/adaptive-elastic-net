# Adaptive Elastic-Net

[![python versions](https://img.shields.io/pypi/pyversions/aenet.svg)](https://pypi.org/project/aenet/)
[![version](https://img.shields.io/pypi/v/aenet.svg)](https://pypi.org/project/aenet/)
[![CI](https://github.com/simaki/adaptive-enet/workflows/CI/badge.svg)](https://github.com/simaki/adaptive-enet/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/simaki/adaptive-enet/branch/main/graph/badge.svg)](https://codecov.io/gh/simaki/adaptive-enet)
[![dl](https://img.shields.io/pypi/dm/aenet)](https://pypi.org/project/aenet/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
model.score(X, y)

model = AdaptiveElasticNetCV().fit(X, y)
model.alpha_
model.predict(X)
model.score(X, y)
```

## Reference

*  Zou, H. and Zhang, H. H. (2009). [On the adaptive elastic net with a diverging number of parameters](https://dx.doi.org/10.1214%2F08-AOS625). Annals
of Statistics 37, 1733-1751.
