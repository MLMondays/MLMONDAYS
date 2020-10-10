
# Written by Dr Daniel Buscombe, Marda Science LLC
# for "ML Mondays", a course supported by the USGS Community for Data Integration
# and the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

## see the blog post here for an explanation: https://dbuscombe-usgs.github.io/MLMONDAYS/blog/2020/10/11/blog-post

from imports import *
from glob import glob

csv_folder = '/media/marda/TWOTB/USGS/DATA/TrainReal/annotations'

csv_files = sorted(glob(csv_folder+os.sep+'*.csv'))

##not an easy file naming convention to deal with ... all strings are forced to have the same length

all_label_data = []; files = []
for f in csv_files:
    dat = np.array(pd.read_csv(f))
    all_label_data.append(dat)
    # get the file name root
    tmp = f.replace('annotations', 'images').replace('.csv','')
    # construct filenames for each annotation
    for i in dat[:,0]:
       if i<10:
          files.append(tmp+os.sep+tmp.split(os.sep)[-1]+'_000000000'+str(i)+'.jpg')
       elif i<100:
          files.append(tmp+os.sep+tmp.split(os.sep)[-1]+'_00000000'+str(i)+'.jpg')
       elif i<1000:
          files.append(tmp+os.sep+tmp.split(os.sep)[-1]+'_0000000'+str(i)+'.jpg')
       elif i<10000:
          files.append(tmp+os.sep+tmp.split(os.sep)[-1]+'_000000'+str(i)+'.jpg')
       elif i<100000:
          files.append(tmp+os.sep+tmp.split(os.sep)[-1]+'_00000'+str(i)+'.jpg')

all_label_data = np.vstack(all_label_data)
files = np.vstack(files).squeeze()

print(all_label_data.shape)
# 87167 annotations, 10 columns
print(files.shape)
# 87167 filenames, 1 column

# so we have converted all the ids to filenames already, next we need to make xmaxs ymaxs
xmax = all_label_data[:,2] + all_label_data[:,4] #xmin + width
ymax = all_label_data[:,3] + all_label_data[:,5] #ymin + height

# list of integers
classes = all_label_data[:,7]
# mapping from integers to strings
class_dict = {-1:'unknown',0: 'human', 1:'elephant', 2:'lion', 3:'giraffe'}
#list of strings
classes_string = [class_dict[i] for i in classes]


d = {'filename': files, 'width': all_label_data[:,4], 'height': all_label_data[:,5], 'class': classes_string,
     'xmin': all_label_data[:,2], 'ymin': all_label_data[:,3], 'xmax': xmax, 'ymax': ymax }
df = pd.DataFrame(data=d)

df.keys()
df.head()
df.tail()

#write to file

df.to_csv('conservationdrones_labels.csv')

# much better - all labels in one file, and easier to read (in fact, it is stand-alone)

###############################################################
## VARIABLES
###############################################################
root = 'data/conservationdrones'+os.sep
output_path = root+'conservationdrones.tfrecord'
csv_input = root+'conservationdrones_labels.csv'


writer = tf.io.TFRecordWriter(output_path)

examples = pd.read_csv(csv_input)
print('Number of labels: %i' % len(examples))
grouped = split(examples, 'filename')

nb_images=len(grouped)
print('Number of images: %i' % nb_images)


# this differs from create_tf_example_coco in that filename paths need not be specified and concatenated

def create_tf_example_conservationdrones(group):
    """
    create_tf_example_conservationdrones(group)
    ""
    This function creates an example tfrecord consisting of an image and label encoded as bytestrings
    The jpeg image is read into a bytestring, and the bbox coordinates and classes are collated and
    converted also
    INPUTS:
        * group [pandas dataframe group object]
        * path [tensorflow dataset]: training dataset
    OPTIONAL INPUTS: None
    OUTPUTS:
        * tf_example [tf.train.Example object]
    GLOBAL INPUTS: BATCH_SIZE
    """
    with tf.io.gfile.GFile(group.filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    filename = group.filename.encode('utf8')

    ids = []
    areas = []
    xmins = [] ; xmaxs = []; ymins = []; ymaxs = []
    labels = []
    is_crowds = []

    #for converting back to integer
    class_dict = {'unknown':-1,'human':0,'elephant':1, 'lion':2, 'giraffe':3}

    for index, row in group.object.iterrows():
        labels.append(class_dict[row['class']])
        ids.append(index)
        xmins.append(row['xmin'])
        ymins.append(row['ymin'])
        xmaxs.append(row['xmax'])
        ymaxs.append(row['ymax'])
        areas.append((row['xmax']-row['xmin'])*(row['ymax']-row['ymin']))
        is_crowds.append(False)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'objects/is_crowd': int64_list_feature(is_crowds),
        'image/filename': bytes_feature(filename),
        'image/id': int64_list_feature(ids),
        'image': bytes_feature(encoded_jpg),
        'objects/xmin': float_list_feature(xmins), #xs
        'objects/xmax': float_list_feature(xmaxs), #xs
        'objects/ymin': float_list_feature(ymins), #xs
        'objects/ymax': float_list_feature(ymaxs), #xs
        'objects/area': float_list_feature(areas), #ys
        'objects/id': int64_list_feature(ids), #ys
        'objects/label': int64_list_feature(labels),
    }))

    return tf_example

for group in grouped:
    tf_example = create_tf_example_conservationdrones(group)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path)
print('Successfully created the TFRecords: {}'.format(output_path))

## this is a big file (1.2 GB)


## create smaller files with 1000 examples per file

ims_per_shard = 1000

SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)
print(SHARDS)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))
print(shared_size)

# create indices into grouped that will enable writing SHARDS files, each containing shared_size examples
grouped_forshards = np.lib.stride_tricks.as_strided(np.arange(len(grouped)), (SHARDS, shared_size))

counter= 0
for indices in grouped_forshards[:-1]:

    tmp = []
    for i in indices:
        tmp.append(grouped[i])

    output_path = root+'conservationdrones.tfrecord'
    output_path = output_path.replace('.tfrecord','')+ "{:02d}-{}.tfrec".format(counter, shared_size)
    writer = tf.io.TFRecordWriter(output_path)

    for group in tmp:
        tf_example = create_tf_example_conservationdrones(group)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))

    counter += 1






filenames = sorted(tf.io.gfile.glob(os.getcwd()+os.sep+root+'*.tfrec'))

features = {
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'objects/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'objects/ymin': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/xmax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/ymax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, features)

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)

import matplotlib.pyplot as plt

for i in dataset.take(10):
    image = tf.image.decode_jpeg(i['image'], channels=1)
    bbox = tf.numpy_function(np.array,[[i["objects/xmin"], i["objects/ymin"], i["objects/xmax"], i["objects/ymax"]]], tf.float32).numpy().T#.squeeze()
    print(len(bbox))

    ids = []
    for id in i["objects/label"].numpy():
       ids.append(class_dict[id])

    fig =plt.figure(figsize=(16,16))
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray)
    ax = plt.gca()

    for box,id in zip(bbox,ids):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=[0, 1, 0], linewidth=1)
        ax.add_patch(patch)
        ax.text(x1, y1, id, bbox={"facecolor": [0, 1, 0], "alpha": 0.4}, clip_box=ax.clipbox, clip_on=True, fontsize=5)
    #plt.show()
    plt.savefig('conservationdrone_label.png',dpi=200, bbox_inches='tight')
    plt.close('all')
