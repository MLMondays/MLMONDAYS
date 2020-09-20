# MLMONDAYS
Materials for the USGS Deep Learning for Image Classification and Segmentation CDI workshop, October 2020, called ML MONDAYS

One module per week. All modules include slides, videos and jupyter notebooks. Each module introduce two topics. A third (more advanced) topic may be introduced using video

Please go to the [project website](https://dbuscombe-usgs.github.io/MLMONDAYS) for more details

## Warning: these resources are not yet finished, so these instructions are for developers only:

### Conda environment workflow

1. Clone the repo:

`git clone --depth 1 https://github.com/dbuscombe-usgs/MLMONDAYS.git`

2. cd to the repo

`cd MLMONDAYS`

3. Create a conda environment

A. Conda housekeeping

`conda clean --all`
`conda update -n base -c defaults conda`

B. Create new `mlmondays` conda environment

We'll create a new conda environment and install packages into it from conda-forge

`conda env create -f mlmondays.yml`

C. Always a good idea to clean up after yourself

`conda clean --all`

4. Test your environment

A. Test 1: does your Tensorflow installation see your GPU?

`python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"`

You should see a bunch of output from tensorflow, and then this line at the bottom

`Num GPUs Available:  1`

B. Test 2: is your jupyter kernel set up correctly?

`jupyter kernelspec list`

should show your python3 kernel inside your `anaconda3/envs/mlmondays` directory, for example

```
Available kernels:
  python3    /home/marda/anaconda3/envs/mlmondays/share/jupyter/kernels/python3
```

C. Change directory to the lesson you wish to work on, e.g. `1_ImageRecog`

`cd 1_ImageRecog`

D. Launch using:

`jupyter notebook`

which should open your browser and show your directory structure within the jupyter environment

To shut down, use `Ctrl+C`


### Google Colab workflow

In each section, there are notebooks you can run on Google Colab instead of your own machine. They are identified by `colab` in the name.

Colab provide free GPU access to run these computational notebooks. If you save the notebooks to your own Google Drive, you can launch them from there.
