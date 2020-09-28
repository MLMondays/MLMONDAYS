# MLMONDAYS
Materials for the USGS Deep Learning for Image Classification and Segmentation CDI workshop, October 2020, called ML MONDAYS

One module per week. All modules include slides, videos and jupyter notebooks.

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


## Useful Links

### Reference

* [Github repository](https://github.com/dbuscombe-usgs/MLMONDAYS) (where all the code and website lives)

* [Documentation](https://dbuscombe-usgs.github.io/MLMONDAYS/docs/doc1)

* [API](https://dbuscombe-usgs.github.io/MLMONDAYS/docs/doc4) (list of all code functions, their inputs and outputs and what they do)

* [Summary of models](https://dbuscombe-usgs.github.io/MLMONDAYS/docs/doc1)

* [Blog](https://dbuscombe-usgs.github.io/MLMONDAYS/blog/)

### Notebooks

* [Part 1a jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageRecog/notebooks/MLMondays_week1_live_partA.ipynb)

* [Part 1a jupyter notebook for Colab](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageRecog/notebooks/MLMondays_week1_live_partA_colab.ipynb)

* [Part 1b jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageRecog/notebooks/MLMondays_week1_live_partB.ipynb)

* [Part 2 jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/2_ObjRecog/notebooks/MLMondays_week2_live.ipynb)

* [Part 2 jupyter notebook for Colab](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/2_ObjRecog/notebooks/MLMondays_week2_live_colab.ipynb)

<!-- * [Part 3 jupyter notebook for Colab](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageSeg/notebooks/MLMondays_week3_live.ipynb)

* [Part 3 jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageSeg/notebooks/MLMondays_week3_live_colab.ipynb) -->

* [Part 4 jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageUnsupRecog/notebooks/MLMondays_week4_live.ipynb)

* [Part 4 jupyter notebook for Colab](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageUnsupRecog/notebooks/MLMondays_week4_live_colab.ipynb)

### Data

* [Part 1 datasets repository](https://github.com/dbuscombe-usgs/mlmondays_data_imrecog)

* [Part 2 datasets repository](https://github.com/dbuscombe-usgs/mlmondays_data_objrecog)

* [Part 3 datasets repository](https://github.com/dbuscombe-usgs/mlmondays_data_imseg)

* [Part 4 datasets repository](https://github.com/dbuscombe-usgs/mlmondays_data_ssimrecog)



## Contents

### Week 1: Image recognition

Get the data by running the download script. Only download the data you wish to use

```
cd 1_ImageRecog
python download_data.py
```

#### notebook lessons (these are the 'live' components of ML-Mondays)
* `notebooks/MLMondays_week1_live_partA.ipynb`: TAMUCC 4-class data visualization
* `notebooks/MLMondays_week1_live_partB.ipynb`: TAMUCC 4-class model building and evaluation

There are also `colab` versions of both notebooks that you can save to your own google drive, then launch in google colab

#### data viz. scripts
* `nwpu_dataviz.py`
* `tamucc_dataviz.py`

#### model training and evaluation scripts
* `tamucc_imrecog_part1a.py`
  * load the subset 2-class (developed/undeveloped) train and validation datasets
  * augment the data
  * make a small custom categorical model
  * train the model with the augmented data
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `tamucc_imrecog_part1b.py`
  * load the subset 2-class (developed/undeveloped) train and validation datasets
  * augment the data
  * make a small custom categorical model and compute class weights
  * train the model using the class weights with the augmented data
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `tamucc_imrecog_part1c.py`
  * load the full 2-class (developed/undeveloped) train and validation datasets
  * make a large custom categorical model
  * train the model
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `tamucc_imrecog_part2a.py`
  * load the subset 3-class (developed/marsh/other) train and validation datasets
  * augment the data
  * make a large custom categorical model
  * train the model with the augmented data
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `tamucc_imrecog_part2b.py`
  * load the subset 3-class (developed/marsh/other) train and validation datasets
  * augment the data
  * make a model based on a mobilenet feature extractor with imagenet weights
  * train the model using the class weights with the augmented data
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `tamucc_imrecog_part2c.py`
  * load the subset 3-class (developed/marsh/other) train and validation datasets
  * augment the data
  * make a model based on a mobilenet feature extractor with imagenet weights
  * load the previous weights, change the learning rate, and freeze the lower layers
  * fine-tune the model with the augmented data
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `tamucc_imrecog_part3a.py`
  * load the subset 4-class train and validation datasets
  * augment the data
  * make a model based on a mobilenet feature extractor with imagenet weights
  * train the model with the augmented data
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `tamucc_imrecog_part3b.py`
  * load the subset 4-class train and validation datasets
  * augment the data
  * make a model based on a mobilenet feature extractor with imagenet weights
  * train the model using the class weights with the augmented data
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `tamucc_imrecog_part3c.py`
  * load the full 4-class train and validation datasets
  * make a model based on a mobilenet feature extractor with imagenet weights
  * train the model
  * examine the history curves
  * evaluate the model and plot a confusion matrix
* `nwpu_imrecog_part1.py`
  * load the subset 11-class train and validation datasets
  * augment the data
  * make a model based on a mobilenet feature extractor with imagenet weights
  * train the model with the augmented data
  * examine the history curves
  * evaluate the model and plot a confusion matrix

#### functions - this is where most of the code is!
* `imports.py`: wrapper imports function, loads the following three sets of functions from:
* `model_funcs.py`: contains functions required for model building, training, evaluation, loss functions, etc
* `tfrecords_funcs.py`: contains functions required for reading and writing tfrecords, and creating datasets, including preprocessing imagery and labels, etc
* `plot_funcs.py`: contains functions required for data inputs and model outputs, etc

#### dataset-specific imports
* `tamucc_imports.py`
* `nwpu_imports.py`

#### file creation
* `nwpu_make_tfrecords.py`: makes NPWU tfrecords from images with class labels in the file name
* `tamucc_make_tfrecords_sample_4class.py`: makes TAMUCC tfrecords from images with class labels in the file name, and reclassify to 4 classes
* `tamucc_make_tfrecords_sample_3class.py`: makes TAMUCC tfrecords from images with class labels in the file name, and reclassify to 3 classes
* `tamucc_make_tfrecords_sample_2class.py`: makes TAMUCC tfrecords from images with class labels in the file name, and reclassify to 2 classes
* `tamucc_make_tfrecords.py`: makes TAMUCC tfrecords from images with class labels in the file name



### Week 2: Object recognition

Get the data and model weights by running the download script.

```
cd 2_ObjRecog
python download_data.py
```

#### notebook lessons (these are the 'live' components of ML-Mondays)
* `notebooks/MLMondays_week2_live.ipynb`

There is also `colab` versions of the notebook that you can save to your own google drive, then launch in google colab

#### model training and evaluation scripts
* `coco_objrecog_part1.py`:
  * view a few examples from the secoora validation dataset
  * make a model, load coco weights and use the model on coco imagery
  * view some examples
  * fine-tune the model on a coco data subset
  * view some examples of fine-tuned model predictions on secoora imagery
* `secoora_objrecog_part1.py`
  * view a few examples from the secoora training dataset
  * make a model, load coco weights and use the model on secoora imagery
  * view some examples
  * fine-tune the model on a secoora data subset
  * view some examples of fine-tuned model predictions on secoora imagery
* `secoora_objrecog_part2.py`
  * view a few examples from the secoora sample jpeg dataset
  * make a model, and train it from scratch on secoora imagery
  * view some examples of model predictions on sample secoora imagery
  * compare the number of people in each frame with model estimates of the same quantity

#### functions - this is where most of the code is!
* `imports.py`: wrapper imports function, loads the following three sets of functions from:
* `model_funcs.py`: contains functions required for model building, training, evaluation, loss functions, etc
* `tfrecords_funcs.py`: contains functions required for reading and writing tfrecords, and creating datasets, including preprocessing imagery and labels, etc
* `plot_funcs.py`: contains functions required for data inputs and model outputs, etc
* `data_funcs.py`: contains functions required for data transformations and other utilities, etc

#### dataset-specific imports
* `coco_imports.py`: imports things like batch size and other fixtures of the model-dataset combo

#### file creation
* `secoora_make_tfrecords.py`: this function creates tfrecords from a csv file containing rows of filename, xmin, ymin, xmax, ymax, and class , and a folder of images


### Week 3: Image segmentation

Get the data by running the download script. Only download the data you wish to use

```
cd 3_ImSeg
python download_data.py
```

#### notebook lessons (these are the 'live' components of ML-Mondays)
* `notebooks/``
* `notebooks/``

There are also `colab` versions of both notebooks that you can save to your own google drive, then launch in google colab

#### data viz. scripts
* ?

#### model training and evaluation scripts
* `obx_imseg_part1.py`
* `obx_imseg_part2.py`
* `obx_imseg_part2b.py`
* `oyster_imseg_part1.py`
* `oyster_imseg_part2.py`

#### functions - this is where most of the code is!
* `imports.py`: wrapper imports function, loads the following three sets of functions from:
* `model_funcs.py`: contains functions required for model building, training, evaluation, loss functions, etc
* `tfrecords_funcs.py`: contains functions required for reading and writing tfrecords, and creating datasets, including preprocessing imagery and labels, etc
* `plot_funcs.py`: contains functions required for data inputs and model outputs, etc


#### dataset-specific imports
* `oyster_imports.py`

#### file creation
* `oysternet_make_tfrecords.py`
* `obx_make_tfrecords.py`


### Week 4: Self-supervised Image recognition

Get the data by running the download script. Only download the data you wish to use

```
cd 4_ImageRecog
python download_data.py
```

#### notebook lessons (these are the 'live' components of ML-Mondays)
* `notebooks/MLMondays_week1_live.ipynb`: TAMUCC 4-class model building and evaluation

There are also `colab` versions of both notebooks that you can save to your own google drive, then launch in google colab

#### model training and evaluation scripts
* `tamucc_imrecog_part1a.py`
* `tamucc_imrecog_part1b.py`
* `tamucc_imrecog_part2.py`
* `tamucc_imrecog_part3.py`
* `tamucc_imrecog_part4.py`
* `nwpu_ssimrecog_part1.py`

#### functions - this is where most of the code is!
* `imports.py`: wrapper imports function, loads the following three sets of functions from:
* `model_funcs.py`: contains functions required for model building, training, evaluation, loss functions, etc
* `tfrecords_funcs.py`: contains functions required for reading and writing tfrecords, and creating datasets, including preprocessing imagery and labels, etc
* `plot_funcs.py`: contains functions required for data inputs and model outputs, etc

#### dataset-specific imports
* `tamucc_imports.py`
* `nwpu_imports.py`

#### file creation
* `tamucc_make_tfrecords_sample_12class.py`



## General workflow using your own data

### Part 1: Supervised Image Recognition

1. Create a TFREcord dataset from your data, organised as follows:
  * copy training images into a folder called `train`
  * copy validation images into a folder called `validation`
  * ensure the class name is written to each file name. Ideally this is a prefix such that it is trivial to extract the class name from the file name
  * modify one of the provided workflows (such as `tamucc_make_tfrecords.py`) for your dataset, to create your train and validation tfrecord shards

2. Set up your model
  * Decide on whether you want to train a small custom model from scratch, a large model from scratch, or a large model trained using weights transfered from another task
  * If a small custom model, use `make_cat_model` with `shallow=True` for a relatively small model, and `shallow=False` for a relatively large model
  * If a large model with transfer learning, decide on which one to utilize (`transfer_learning_mobilenet_model`, `transfer_learning_xception_model`, or `transfer_learning_model_vgg`)
  * If you wish to train a large model from scratch, decide on which one to utilize (`mobilenet_model`, or `xception_model`)

3. Set up a data pipeline
  * Modify and follow the provided examples to create a `get_training_dataset()` and `get_validation_dataset()`. This will likely require you copy and modify `get_batched_dataset` to your own needs, depending on the format of your labels in filenames, by writing your own `read_tfrecord` function for your dataset (depending on the model selected)

4. Set up a model training pipeline
  * `.compile()` your model with an appropriate loss function and metrics
  * define a `LearningRateScheduler` function to vary learning rates over training as a function of training epoch
  * define an `EarlyStopping` criteria and create a `ModelCheckpoint` to save trained model weights
  * if transfer learning using weights not from imagenet, load your initial weights from somewhere else

5. Train the model
  * Use `history = model.fit()` to create a record of the training history. Pass the training and validation datasets, and a list of callbacks containing your model checkpoint, learning rate scheduler, and early stopping monitor)

6. Evaluate your model
  * Plot and study the `history` time-series of losses and metrics. If unsatisfactory, begin the iterative process of model optimization  
  * Use the `loss, accuracy = model.evaluate(get_validation_dataset(), batch_size=BATCH_SIZE, steps=validation_steps)` function using the validation dataset and specifying the number of validation steps
  * Make plots of model outputs, organized in such a way that you can at-a-glance see where the model is failing. Make use of `make_sample_plot` and `p_confmat`, as a starting point, to visualize sample imagery with their model predictions, and a confusion matrix of predicted/true class-correspondences


### Part 2: Object Recognition

1. Create a TFREcord dataset from your data, organised as follows:
  * copy training images into a folder called `train`
  * copy validation images into a folder called `validation`
  * create a text, csv file that lists each of the objects in each image, with the following columns: filename, xmin, ymin, xmax (float y coord pixel), ymax (float y coord pixel), class (string)
  * modify the provided workflow (`secoora_make_tfrecords.py`) for your dataset, to create your train and validation tfrecord shards

```
  filename,	width,	height,	class,	xmin,	ymin,	xmax,	ymax
  staugustinecam.2019-04-18_1400.mp4_frames_25.jpg,	1280,	720,	person,	1088,	581,	1129,	631
  staugustinecam.2019-04-18_1400.mp4_frames_25.jpg,	1280,	720,	person,	1125,	524,	1183,	573
  staugustinecam.2019-04-04_0700.mp4_frames_51.jpg,	1280,	720,	person,	158,	198,	178,	244
  staugustinecam.2019-04-04_0700.mp4_frames_51.jpg,	1280,	720,	person,	131,	197,	162,	244
  staugustinecam.2019-04-04_0700.mp4_frames_51.jpg,	1280,	720,	person,	40,	504,	87,	581
  staugustinecam.2019-04-04_0700.mp4_frames_51.jpg,	1280,	720,	person,	0,	492,	15,	572
  staugustinecam.2019-01-01_1400.mp4_frames_44.jpg,	1280,	720,	person,	1086,	537,	1130,	615
  staugustinecam.2019-01-01_1400.mp4_frames_44.jpg,	1280,	720,	person,	1064,	581,	1134,	624
  staugustinecam.2019-01-01_1400.mp4_frames_44.jpg,	1280,	720,	person,	1136,	526,	1186,	570
```

2. Set up your model
  * Decide on whether you want to train a model from scratch, or trained using weights transfered from another task (such as coco 2017)

3. Set up a model training pipeline
  * `.compile()` your model with an appropriate loss function and metrics
  * define a `LearningRateScheduler` function to vary learning rates over training as a function of training epoch
  * define an `EarlyStopping` criteria and create a `ModelCheckpoint` to save trained model weights
  * if transfer learning using weights not from coco, load your initial weights from somewhere else

5. Train the model
  * Use `history = model.fit()` to create a record of the training history. Pass the training and validation datasets, and a list of callbacks containing your model checkpoint, learning rate scheduler, and early stopping monitor)

6. Evaluate your model
  * Plot and study the `history` time-series of losses and metrics. If unsatisfactory, begin the iterative process of model optimization  
  * Use the `loss, accuracy = model.evaluate(get_validation_dataset(), batch_size=BATCH_SIZE, steps=validation_steps)` function using the validation dataset and specifying the number of validation steps
  * Make plots of model outputs, organized in such a way that you can at-a-glance see where the model is failing. Make use of `visualize_detections`, as a starting point, to visualize sample imagery with their model predictions


### Part 4: Semi-supervised Image Recognition

1. Create a TFREcord dataset from your data, organised as follows:
  * Copy training images into a folder called `train`
  * Copy validation images into a folder called `validation`
  * Ensure the class name is written to each file name. Ideally this is a prefix such that it is trivial to extract the class name from the file name
  * Modify one of the provided workflows (such as `tamucc_make_tfrecords.py`) for your dataset, to create your train and validation tfrecord shards

2. Set up your model
  * Decide on whether you want to train a small or large embedding model (`get_embedding_model` or `get_large_embedding_model`)

3. Set up a data pipeline
  * Modify and follow the provided examples to create a `get_training_dataset()` and `get_validation_dataset()`. This will likely require you copy and modify `get_batched_dataset` to your own needs, depending on the format of your labels in filenames, by writing your own `read_tfrecord` function for your dataset (depending on the model selected)
  * Remember for this method you have to read all the data at once into memory, which isn't ideal. You may therefore need to modify `get_data_stuff` to be a more efficient way to do so for your data

4. Set up a model training pipeline
  * `.compile()` your model with an appropriate loss function and metrics
  * Define a `LearningRateScheduler` function to vary learning rates over training as a function of training epoch
  * Define an `EarlyStopping` criteria and create a `ModelCheckpoint` to save trained model weights

5. Train the autoencoder embedding model
  * Use `history = model.fit()` to create a record of the training history. Pass the training and validation datasets, and a list of callbacks containing your model checkpoint, learning rate scheduler, and early stopping monitor)

6. Train the k-nearest neighbour classifer
  * Decide or determine the optimal number of neighbours (`k`)
  * Use `fit_knn_to_embeddings` to make a model of your training embeddings

6. Evaluate your model
  * Plot and study the `history` time-series of losses and metrics. If unsatisfactory, begin the iterative process of model optimization  
  * Use the `loss, accuracy = model.evaluate(get_validation_dataset(), batch_size=BATCH_SIZE, steps=validation_steps)` function using the validation dataset and specifying the number of validation steps
  * Make plots of model outputs, organized in such a way that you can at-a-glance see where the model is failing. Make use of `make_sample_plot` and `p_confmat`, as a starting point, to visualize sample imagery with their model predictions, and a confusion matrix of predicted/true class-correspondences
  * On the test set, play `tf.nn.l2_normalize` (i.e. don't use it on test and/or train embeddings and see if it improves results)
