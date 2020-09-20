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

from imports import *

###############################################################
## VARIABLES
###############################################################

imdir = '/media/marda/TWOTB/USGS/DATA/OysterNet/1kx1k_dataset/all_images'

# Convert folder of pngs into jpegs
# for file in *.png
# > do
# > convert $file $"${file%.png}.jpg"
# > done

lab_path = '/media/marda/TWOTB/USGS/DATA/OysterNet/1kx1k_dataset/all_labels'

tfrecord_dir = '/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/3_ImageSeg/data/oysternet'

images = tf.io.gfile.glob(imdir+os.sep+'*.jpg')

images = tf.io.gfile.glob(lab_path+os.sep+'*.jpg')


###############################################################
## EXECUTION
###############################################################
nb_images=len(tf.io.gfile.glob(imdir+os.sep+'*.jpg'))

SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))

dataset = get_seg_dataset_for_tfrecords(imdir, lab_path, shared_size)

## view a batch
# for imgs,lbls in dataset.take(1):
#   imgs = imgs[:BATCH_SIZE]
#   lbls = lbls[:BATCH_SIZE]
#   for count,(im,lab) in enumerate(zip(imgs,lbls)):
#      plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
#      plt.imshow(tf.image.decode_jpeg(im, channels=3))
#      plt.imshow(tf.image.decode_jpeg(lab, channels=1), alpha=0.5, cmap='gray')
#      plt.axis('off')
# plt.show()


write_seg_records(dataset, tfrecord_dir)

#
