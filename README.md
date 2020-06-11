# AR-Net
A simple Auto-Regressive Neural Network for time-series ([link to paper](https://arxiv.org/abs/1911.12436)).

## Versions
### 0.1
Version 0.1 was based on Pytorch and you can still use it if you do not like fastai. Seejust file [`v0_1/example.py`](v0_1/example.py) for how to use the v0.1 model.
### 1.0
The new version 1.0 makes the model easier to use with your own dataset and requires less hyperparameters for a simpler training procedure. It is built on the latest release of Fastai2. View the notebook [`v1_0/example_ar_net.ipynb`](v1_0/example_ar_net.ipynb) for an example of how to use the model.
### New
AR-Net is now part of a more comprehensive package [NeuralProphet](https://github.com/ourownstory/neural_prophet). I strongly recommend using it instead of the standalone version, unless you need to model a highly-autoregressive time-series with sparse long-range dependencies.

## Notes
- My pull request in fastai2 was merged and has made it's way to the current pypi package (``fastai2==0.0.16``), so you can now  install fastai2 directly with pip. Note: other versions than 0.0.16 might break.
- Other dependencies include ``numpy, pandas, matplotlib, seaborn, pytorch``.
