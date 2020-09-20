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



## Contents

### Week 1: Image recognition

### notebook lessons (these are the 'live' components of ML-Mondays)
* notebooks/MLMondays_week1_live_partA.ipynb: TAMUCC 4-class data visualization
* notebooks/MLMondays_week1_live_partB.ipynb: TAMUCC 4-class model building and evaluation

There are also `colab` versions of both notebooks that you can save to your own google drive, then launch in google colab

#### data viz. scripts
* nwpu_dataviz.py
* tamucc_dataviz.py

#### model training and evaluation scripts
* tamucc_imrecog_part1a.py
* tamucc_imrecog_part1b.py
* tamucc_imrecog_part1c.py
* tamucc_imrecog_part3a.py
* tamucc_imrecog_part3b.py
* tamucc_imrecog_part3c.py
* tamucc_imrecog_part2a.py
* tamucc_imrecog_part2b.py
* tamucc_imrecog_part2c.py
* nwpu_imrecog_part1.py

#### functions - this is where most of the code is!
* imports.py

#### dataset-specific imports
* tamucc_imports.py
* nwpu_imports.py

#### file creation
* nwpu_make_tfrecords.py
* tamucc_make_tfrecords_sample_4class.py
* tamucc_make_tfrecords_sample_3class.py
* tamucc_make_tfrecords_sample_2class.py
* tamucc_make_tfrecords.py



### Week 4: Self-supervised Image recognition

### notebook lessons (these are the 'live' components of ML-Mondays)
* notebooks/MLMondays_week1_live.ipynb: TAMUCC 4-class model building and evaluation

There are also `colab` versions of both notebooks that you can save to your own google drive, then launch in google colab

#### model training and evaluation scripts
* tamucc_imrecog_part1a.py
* tamucc_imrecog_part1b.py
* tamucc_imrecog_part2.py
* tamucc_imrecog_part3.py
* tamucc_imrecog_part4.py
* nwpu_ssimrecog_part1.py

#### functions - this is where most of the code is!
* imports.py

#### dataset-specific imports
* tamucc_imports.py
* nwpu_imports.py

#### file creation
* tamucc_make_tfrecords_sample_12class.py
