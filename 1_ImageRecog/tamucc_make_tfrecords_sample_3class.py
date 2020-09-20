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
###############################################################
## IMPORTS
###############################################################

from imports import *


#============================================

imdir = '/media/marda/TWOTB/USGS/DATA/tamucc_coastal_imagery/subset'
csvfile = '/media/marda/TWOTB/USGS/DATA/tamucc_coastal_imagery/tamucc_subset.csv'
recoded_dir = '/media/marda/TWOTB/USGS/DATA/tamucc_coastal_imagery/subset_recoded_3class'
# os.mkdir(recoded_dir)
tfrecord_dir = '1_ImageRecog/data/tamucc/subset_3class/'+str(TARGET_SIZE)
# os.mkdir(tfrecord_dir)

dat = pd.read_csv(csvfile)
CLASSES = np.unique(dat['class'].values)
print(CLASSES)

CLASSES = [c.encode() for c in CLASSES]

#low energy
marsh_classes = [c for c in CLASSES if b'marsh' in c] + [c for c in CLASSES if b'swamp' in c] + [c for c in CLASSES if b'flat' in c]

#developed
dev_classes = [c for c in CLASSES if b'structures' in c] + [c for c in CLASSES if b'manmade' in c]

#high energy
other_classes = np.setdiff1d(np.setdiff1d(CLASSES, dev_classes), marsh_classes).tolist()



# for f,c in zip(dat.file.values, dat['class'].values):
#   if c.encode() in marsh_classes:
#      shutil.copy(imdir+os.sep+f, recoded_dir+os.sep+'marsh'+'_'+f)
#   elif c.encode() in dev_classes:
#      shutil.copy(imdir+os.sep+f, recoded_dir+os.sep+'dev'+'_'+f)
#   else:
#      shutil.copy(imdir+os.sep+f, recoded_dir+os.sep+'other'+'_'+f)


im, lab = read_image_and_label(recoded_dir+os.sep+'dev_IMG_0158_SecMO_Sum12_Pt3.jpg')
print(lab)

im, lab = read_image_and_label(recoded_dir+os.sep+'other_IMG_1576_SecBC_Spr12.jpg')
print(lab)

im, lab = read_image_and_label(recoded_dir+os.sep+'marsh_IMG_4882_SecJMO_Sum12_Pt3.jpg')
print(lab)


nb_images=len(tf.io.gfile.glob(recoded_dir+os.sep+'*.jpg'))

SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)
print(SHARDS)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))
print(shared_size)

tamucc_dataset = get_dataset_for_tfrecords(recoded_dir, shared_size)

write_records(tamucc_dataset, tfrecord_dir)

#
# tamucc_dataset = tf.data.Dataset.list_files(recoded_dir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
# tamucc_dataset = tamucc_dataset.map(read_image_and_label)
# tamucc_dataset = tamucc_dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)
#
# tamucc_dataset = tamucc_dataset.map(recompress_image, num_parallel_calls=AUTO)
# tamucc_dataset = tamucc_dataset.batch(shared_size)
#
# CLASSES = [b'marsh', b'dev', b'other']
#
# for shard, (image, label) in enumerate(tamucc_dataset):
#   shard_size = image.numpy().shape[0]
#   filename = tfrecord_dir+os.sep+"tamucc" + "{:02d}-{}.tfrec".format(shard, shard_size)
#
#   with tf.io.TFRecordWriter(filename) as out_file:
#     for i in range(shard_size):
#       example = to_tfrecord(image.numpy()[i],label.numpy()[i], CLASSES)
#       out_file.write(example.SerializeToString())
#     print("Wrote file {} containing {} records".format(filename, shard_size))
