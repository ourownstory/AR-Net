# AR-Net
A simple Auto-Regressive Neural Network for time-series.

Read [the AR-Net paper](https://arxiv.org/abs/1911.12436).

Version 1.0 makes the model easier to use with your own dataset and requires less hyperparameters for a simpler training procedure. It is built on the latest release of Fastai2. View the notebook `v1_0/example_ar_net.ipynb` for an example of how to use the model.

Make sure you install the latest version direclty from [their github](https://github.com/fastai/fastai2#installing). (I had to extend their code, but my pull request got promply merged.)

Version 0.1 was based on Pytorch and you can still use it if you do not like fastai. See file `v0_1/example.py` for how to use the v0.1 model.
