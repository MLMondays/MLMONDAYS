
###############################################################
## IMPORTS
###############################################################
from imports import *

###############################################################
### FUNCTIONS
###############################################################

#-----------------------------------
def get_training_dataset():
    """
    This function will return a batched dataset for model training
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: training_filenames
    OUTPUTS: batched data set object
    """
    return get_batched_dataset(training_filenames)

#-----------------------------------
def get_validation_dataset():
    """
    This function will return a batched dataset for model validation
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: validation_filenames
    OUTPUTS: batched data set object
    """
    return get_batched_dataset(validation_filenames)

#-----------------------------------
def compute_hist(images):
    """
    Compute the per channel histogram for a batch
    of images
    INPUTS:
        * images [ndarray]: batch of shape (N x W x H x 3)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * hist_r [dict]: histogram frequencies {'hist'} and bins {'bins'} for red channel
        * hist_g [dict]: histogram frequencies {'hist'} and bins {'bins'} for green channel
        * hist_b [dict]: histogram frequencies {'hist'} and bins {'bins'} for blue channel
    """
    images = images/255.
    mean = np.mean(images, axis=0, dtype=np.float64)

    mean_r, mean_g, mean_b = mean[:,:,0], mean[:,:,1], mean[:,:,2]
    mean_r = np.reshape(mean_r, (-1, 1))
    mean_g = np.reshape(mean_g, (-1, 1))
    mean_b = np.reshape(mean_b, (-1, 1))

    hist_r_, bins_r = np.histogram(mean_r, bins="auto")
    hist_g_, bins_g = np.histogram(mean_g, bins="auto")
    hist_b_, bins_b = np.histogram(mean_b, bins="auto")

    hist_r = {"hist": hist_r_, "bins": bins_r}
    hist_g = {"hist": hist_g_, "bins": bins_g}
    hist_b = {"hist": hist_b_, "bins": bins_b}

    return hist_r, hist_g, hist_b

#-----------------------------------
def plot_distribution(images, labels, class_id):
    """
    Compute the per channel histogram for a batch
    of images
    INPUTS:
        * images [ndarray]: batch of shape (N x W x H x 3)
        * labels [ndarray]: batch of shape (N x 1)
        * class_id [int]: class integer to plot
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: matplotlib figure
    """
    fig = plt.figure(figsize=(21,7))
    rows, cols = 1, 3
    locs = np.where(labels == class_id)
    samples = locs[:][0]
    class_images = images[samples]
    hist_r, hist_g, hist_b = compute_hist(class_images)
    plt.title("Histogram - Mean Pixel Value:  " + CLASSES[class_id])
    plt.axis('off')

    fig.add_subplot(rows, cols, 1)
    hist, bins = hist_r["hist"], hist_r["bins"]
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width,color='r')
    plt.xlim((0,1))
    plt.ylim((0, 255))

    fig.add_subplot(rows, cols, 2)
    hist, bins = hist_g["hist"], hist_g["bins"]
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width,color='g')
    plt.xlim((0,1))
    plt.ylim((0,255))

    fig.add_subplot(rows, cols, 3)
    hist, bins = hist_b["hist"], hist_b["bins"]
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width,color='b')
    plt.xlim((0,1))
    plt.ylim((0, 255))

#-----------------------------------
def plot_one_class(inp_batch, sample_idx, label, batch_size, rows=8, cols=8, size=(20,15)):
    """
    Plot "batch_size" images that belong to the class "label"
    INPUTS:
        * inp_batch
        * sample_idx
        * label
        * batch_size
    OPTIONAL INPUTS:
        * rows=8
        * cols=8
        * size=(20,15)
    GLOBAL INPUTS: matplotlib figure
    """
    OUTPUTS:
    fig = plt.figure(figsize=size)
    plt.title(CLASSES[int(label)])
    plt.axis('off')
    for n in range(0, batch_size):
        fig.add_subplot(rows, cols, n + 1)
        img = inp_batch[n]
    plt.imshow(img)
    plt.axis('off')

#-----------------------------------
def compute_mean_image(images, opt="mean"):
    """
    Compute and return mean image given
    a batch of images
    INPUTS:
        * images [ndarray]: batch of shape (N x W x H x 3)
    OPTIONAL INPUTS:
        * opt="mean" or "median"
    GLOBAL INPUTS:
    OUTPUTS: 2d mean image [ndarray]
    """
    images = images/255.
    if opt == "mean":
        return np.mean(images, axis=0, dtype=np.float64)
    else:
        return np.median(images, axis=0)

#-----------------------------------
def plot_mean_images(images, labels):
    """
    Plot the mean image of a set of images
    INPUTS:
        * images [ndarray]: batch of shape (N x W x H x 3)
        * labels [ndarray]: batch of shape (N x 1)
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: matplotlib figure
    """
    fig = plt.figure(figsize=(20,15))
    rows, cols = 1, 3
    example_images = []
    for n in np.arange(len(CLASSES)):
        fig.add_subplot(rows, cols, n + 1)
        locs = np.where(labels == n)
        samples = locs[:][0]
        class_images = images[samples]
        img = compute_mean_image(class_images, "median")
        plt.imshow(img)
        plt.title(CLASSES[n])
        plt.axis('off')

#-----------------------------------
def plot_tsne(tsne_result, label_ids):
    """
    Plot TSNE loadings and colour code by class
    Source: https://www.kaggle.com/gaborvecsei/plants-t-sne
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: matplotlib figure, matplotlib figure axes object
    """
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111,projection='3d')

    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
      ax.scatter(tsne_result[np.where(label_ids == label_id), 0],
                  tsne_result[np.where(label_ids == label_id), 1],
                  tsne_result[np.where(label_ids == label_id), 2],
                  alpha=0.8,
                  color= plt.cm.Set1(label_id / float(nb_classes)),
                  marker='o',
                  label=CLASSES[label_id])
    ax.legend(loc='best')
    ax.axis('tight')

    ax.view_init(25, 45)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    return fig, ax

#-----------------------------------
# Show images with t-SNE
# Source: https://www.kaggle.com/gaborvecsei/plants-t-sne
def visualize_scatter_with_images(X_2d_data, images, figsize=(15,15), image_zoom=1):
    """
    Plot TSNE loadings and colour code by class
    Source: https://www.kaggle.com/gaborvecsei/plants-t-sne
    INPUTS:
        * X_2d_data
        * images
    OPTIONAL INPUTS:
        * figsize=(15,15)
        * image_zoom=1
    GLOBAL INPUTS:
    OUTPUTS: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0, _ = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data[:,:2])
    ax.autoscale()
    ax.axis('tight')
    return fig


####================================================

data_path= os.getcwd()+os.sep+"data/tamucc/subset_2class/400"

training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))

CLASSES = ['undev', 'dev']

nb_images = ims_per_shard * len(training_filenames)
print(nb_images)


num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)
X_train = []
ytrain = []
train_ds = get_training_dataset()
for imgs,lbls in train_ds.take(num_batches):
  n = np.bincount(lbls, minlength=len(CLASSES))
  ytrain.append(lbls.numpy())
  for im in imgs:
    X_train.append(im)


X_train = np.array(X_train)

ytrain = np.hstack(ytrain)


# show examples per class

bs = 64
for class_idx in [0,1,2]:
  #show_one_class(class_idx=class_idx, bs=64)
  locs = np.where(ytrain == class_idx)
  samples = locs[:][0]
  #random.shuffle(samples)
  samples = samples[:bs]
  print("Total number of {} (s) in the dataset: {}".format(CLASSES[class_idx], len(locs[:][0])))
  X_subset = X_train[samples]
  plot_one_class(X_subset, samples, class_idx, bs)
  plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_2class_samples_'+CLASSES[class_idx]+'.png', dpi=200, bbox_inches='tight')
  plt.close('all')



# plot mean images per class

plot_mean_images(X_train, ytrain)
plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_2class_mean.png', dpi=200, bbox_inches='tight')
plt.close('all')


#### plot histograms


plot_distribution(X_train, ytrain, 0)
plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_2class_hist_'+CLASSES[0]+'.png', dpi=200, bbox_inches='tight')
plt.close('all')

plot_distribution(X_train, ytrain, 1)
plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_2class_hist_'+CLASSES[1]+'.png', dpi=200, bbox_inches='tight')
plt.close('all')

plot_distribution(X_train, ytrain, 2)
plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_2class_hist_'+CLASSES[2]+'.png', dpi=200, bbox_inches='tight')
plt.close('all')



num_samples = 1200
X_subset = X_train[:num_samples]
X_subset = X_subset.reshape(num_samples,-1)
y_subset = ytrain[:num_samples]



num_components=100
pca = PCA(n_components=num_components)
reduced = pca.fit_transform(X_subset)
print('Cumulative variance explained by {} principal components: {}'.format(num_components, np.sum(pca.explained_variance_ratio_)))



# Create animation
tsne = TSNE(n_components=3,n_jobs=-1)
tsne_result = tsne.fit_transform(reduced)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)


fig, ax = plot_tsne(tsne_result_scaled,y_subset)
plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_2class_tsne_sample.png', dpi=200, bbox_inches='tight')
plt.close('all')




f = visualize_scatter_with_images(tsne_result_scaled, images = [np.reshape(i, (TARGET_SIZE,TARGET_SIZE,3)) for i in X_subset], image_zoom=0.1)

plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_2class_tsne_vizimages_sample.png', dpi=200, bbox_inches='tight')
plt.close('all')





####================================================

data_path=  os.getcwd()+os.sep+"data/tamucc/subset/400"

training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))

json_file =  os.getcwd()+os.sep+'data/tamucc/subset/tamucc_subset.json'

with open(json_file) as f:
    class_dict = json.load(f)

# string names
CLASSES = [class_dict[k] for k in class_dict.keys()]

nb_images = ims_per_shard * len(training_filenames)
print(nb_images)

len(CLASSES)

num_batches = int(( nb_images) / BATCH_SIZE)
print(num_batches)
X_train = []
ytrain = []
train_ds = get_training_dataset()
for imgs,lbls in train_ds.take(num_batches):
  n = np.bincount(lbls, minlength=len(CLASSES))
  ytrain.append(lbls.numpy()) #n)
  for im in imgs:
    X_train.append(im)

X_train = np.array(X_train)

ytrain = np.hstack(ytrain)


# show examples per class
CLASSES2 = []
bs = 64
for class_idx in np.arange(len(CLASSES)):

  locs = np.where(ytrain == class_idx)
  samples = locs[:][0]
  if len(samples)>64:
      CLASSES2.append(CLASSES[class_idx])
      samples = samples[:bs]
      print("Total number of {} (s) in the dataset: {}".format(CLASSES[class_idx], len(locs[:][0])))
      X_subset = X_train[samples]
      plot_one_class(X_subset, samples, class_idx, bs)
      plt.savefig('tamucc_sample_16class_samples_'+CLASSES[class_idx]+'.png', dpi=200, bbox_inches='tight')
      plt.close('all')
  else:
      print('insufficient imagery: {} ({})'.format(CLASSES[class_idx], len(samples)))

# only take the images from CLASSES2

X_train = []
ytrain = []
train_ds = get_training_dataset()
for imgs,lbls in train_ds.take(num_batches):
    lbls_str = [CLASSES[l] for l in lbls.numpy()]
    ind = np.where([l for l in lbls_str if l in CLASSES2])[0]
    lbls = lbls.numpy()[ind]
    imgs = imgs.numpy()[ind]

    ytrain.append(lbls)
    for im in imgs:
        X_train.append(im)

X_train = np.array(X_train)

ytrain = np.hstack(ytrain)




num_samples = 1200
X_subset = X_train[:num_samples]
X_subset = X_subset.reshape(num_samples,-1)
y_subset = ytrain[:num_samples]



num_components=100
pca = PCA(n_components=num_components)
reduced = pca.fit_transform(X_subset)
print('Cumulative variance explained by {} principal components: {}'.format(num_components, np.sum(pca.explained_variance_ratio_)))



# Create plot
tsne = TSNE(n_components=3,n_jobs=-1)
tsne_result = tsne.fit_transform(reduced)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)


fig, ax = plot_tsne(tsne_result_scaled,y_subset)
plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_12class_tsne_sample.png', dpi=200, bbox_inches='tight')
plt.close('all')




f = visualize_scatter_with_images(tsne_result_scaled, images = [np.reshape(i, (TARGET_SIZE,TARGET_SIZE,3)) for i in X_subset], image_zoom=0.1)

plt.savefig( os.getcwd()+os.sep+'results/tamucc_sample_12class_tsne_vizimages_sample.png', dpi=200, bbox_inches='tight')
plt.close('all')
