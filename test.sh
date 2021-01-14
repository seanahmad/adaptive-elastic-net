#!/bin/sh

python3 -m pytest --doctest-modules adaptive_enet
python3 -m pytest --doctest-modules tests

python3 -m flake8 adaptive_enet
python3 -m black --check adaptive_enet || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m black adaptive_enet
python3 -m isort --check --force-single-line-imports adaptive_enet || read -p "Run formatter? (y/N): " yn; [[ $yn = [yY] ]] && python3 -m isort --force-single-line-imports adaptive_enet
