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

from imageio import imwrite

###############################################################
## VARIABLES
###############################################################

imdir = '/media/marda/TWOTB/USGS/SOFTWARE/mlmondays_prep/imseg/obx/images'

# Convert folder of pngs into jpegs
# for file in *.png
# > do
# > convert $file $"${file%.png}.jpg"
# > done

lab_path = '/media/marda/TWOTB/USGS/SOFTWARE/mlmondays_prep/imseg/obx/labels'

tfrecord_dir = '/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/3_ImageSeg/data/obx'



NY = 7360
NX = 4912

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=5,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

#cant read all images into memory so I'll have to do this in batches
i=41
for k in range(14):

    #set a different seed each time to get a new batch of ten
    seed = int(np.random.randint(0,100,size=1))
    img_generator = image_datagen.flow_from_directory(
            imdir,
            target_size=(NX, NY),
            batch_size=10,
            class_mode=None, seed=seed, shuffle=True)

   #the seed must be the same as for the training set to get the same images
    mask_generator = mask_datagen.flow_from_directory(
            lab_path,
            target_size=(NX, NY),
            batch_size=10,
            class_mode=None, seed=seed, shuffle=True)

    #The following merges the two generators (and their flows) together:
    train_generator = (pair for pair in zip(img_generator, mask_generator))

    #grab a batch of 10 images and label images
    x, y = next(train_generator)

    # wrute them to file and increment the counter
    for im,lab in zip(x,y):
        imwrite(imdir+os.sep+'augimage_000'+str(i)+'.jpg', im)
        imwrite(lab_path+os.sep+'auglabel_000'+str(i)+'_deep_whitewater_shallow_no_water_label.jpg', lab)
        i += 1

    #save memory
    del x, y, im, lab
    #get a new batch



###############################################################
## EXECUTION
###############################################################

nb_images=len(tf.io.gfile.glob(imdir+os.sep+'*.jpg'))

#there are only 146 images total, so we use a smaller number for images per shard
ims_per_shard = 20

SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))

dataset = get_seg_dataset_for_tfrecords_obx(imdir, lab_path, shared_size)

# view a batch
# for imgs,lbls in dataset.take(1):
#   imgs = imgs[:BATCH_SIZE]
#   lbls = lbls[:BATCH_SIZE]
#   for count,(im,lab) in enumerate(zip(imgs,lbls)):
#      plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
#      plt.imshow(tf.image.decode_jpeg(im, channels=3))
#      plt.imshow(tf.image.decode_jpeg(lab, channels=1), alpha=0.5, cmap='gray')
#      plt.axis('off')
# plt.show()

write_seg_records_obx(dataset, tfrecord_dir)

#
