# AR-Net
A simple auto-regressive Neural Network for time-series ([link to paper](https://arxiv.org/abs/1911.12436)).

## Install
After downloading the code repository (via `git clone`), change to the repository directory (`cd AR-Net`) 
and install arnet as python package with `pip install .`

## Use
View the notebook [`arnet/example_ar_net.ipynb`](arnet/example_ar_net.ipynb) for an example of how to use the model.

## Versions
### Now also part of NeuralProphet
AR-Net is now part of a more comprehensive package [NeuralProphet](https://github.com/ourownstory/neural_prophet). 
I strongly recommend using it instead of the standalone version, unless you specifically want to use AR-Net, 
which may make sense if you need to model a highly-autoregressive time-series with sparse long-range dependencies.

### Current (1.1)
The version 1.0 made the model easier to use with your own datasets and requires less hyperparameters 
for a simpler training procedure. It is built on the fastai library.

Changes (1.0 -> 1.1):
* port [beta fastai2](https://github.com/fastai/fastai2) to it's current [stable release](https://github.com/fastai/fastai) 
* make install as pip package possible
* add black code formatter (and git pre-commit hook)
* add unittests (and git pre-push hook)
* fix issues with new fastai api
* remove old code fragments

### Pure PyTorch (0.1)
Version 0.1 was based on Pytorch and you can still use it if you do not want to use fastai. 

See file [`v0_1/example.py`](v0_1/example.py) for how to use the v0.1 model.
