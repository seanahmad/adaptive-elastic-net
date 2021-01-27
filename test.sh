#!/bin/sh -eu

python3 -m pytest --doctest-modules aenet
python3 -m pytest --doctest-modules tests

python3 -m flake8 aenet
python3 -m black --check aenet || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m black aenet
python3 -m isort --check --force-single-line-imports aenet || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m isort --force-single-line-imports aenet
