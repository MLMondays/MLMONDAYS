## Installation

### Conda housekeeping

`conda clean --all`
`conda update -n base -c defaults conda`

## Create new `mlmondays` conda environment

We'll create a new conda environment and install packages into it from conda-forge

`conda env create -f mlmondays.yml`

Always a good idea to clean up after yourself

`conda clean --all`

## Test your environment

### Test 1: does your Tensorflow installation see your GPU?

`python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"`

You should see a bunch of output from tensorflow, and then this line at the bottom

`Num GPUs Available:  1`

### Test 2: is your jupyter kernel set up correctly?

`jupyter kernelspec list`

should show your python3 kernel inside your `anaconda3/envs/mlmondays` directory, for example

```
Available kernels:
  python3    /home/marda/anaconda3/envs/mlmondays/share/jupyter/kernels/python3
```

Launch using:

`jupyter notebook`

which should open your browser and show your directory structure within the jupyter environment

To shut down, use `Ctrl+C`
