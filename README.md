# FLOP
FLOP [1] is a Federated Learning framework, which allows training a neural network in a distributed and privacy-preserving manner.
This repository contains an experiment on the accuracy of models trained with FLOP.

# Experiment
In `main.py` a simulation of the FLOP framework is provided, that simulates training a CNN classifier on the fashion MNIST dataset on 10 simulated clients.

# How to reproduce
To reproduce this experiment you will need to install the following:
1. Python 3.8
2. conda
3. pip

To install all the dependencies run:
```
conda env create --file environment.yml --name flop-env
```

Then, to run the experiment, execute:
```
conda activate flop-env
python main.py
```

The output of the experiment consists of two plots that present the average accuracy and loss of a client model in each epoch.

[1] Qian Yang et al.FLOP: Federated Learning on Med-ical  Datasets  using  Partial  Networks.  2021.  arXiv:2102.05218[cs.LG].
