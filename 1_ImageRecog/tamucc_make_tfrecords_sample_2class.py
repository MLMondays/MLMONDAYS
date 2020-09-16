

###############################################################
## IMPORTS
###############################################################

from imports import *


#============================================


#####################################

# #developed
# class0 = ['coarsegrained_sand_beaches'
#  'exposed_tidal_flats'
#  'finegrained_sand_beaches'
#  'freshwater_marshes_herbaceous_vegetation'
#  'freshwater_swamps_woody_vegetation'
#  'gravel_shell_beaches'
#  'mixed_sand_gravel_shell_beaches'
#  'salt_brackish_water_marshes'
#  'scarps_steep_slopes_clay'
#  'scarps_steep_slopes_sand'
#  'sheltered_scarps'
#  'sheltered_tidal_flats']
#
# #undeveloped
# class1 = [
# 'exposed_riprap_structures'
#  'exposed_walls_other_structures'
#  'sheltered_riprap_structures'
#  'sheltered_solid_manmade'
# ]

dev_classes = [c for c in CLASSES if b'structures' in c] + [c for c in CLASSES if b'manmade' in c]

undev_classes = np.setdiff1d(CLASSES, dev_classes).tolist()

# recoded_dir = '/media/marda/TWOTB/USGS/DATA/tamucc_coastal_imagery/full_recoded_2class'
# # os.mkdir(recoded_dir)
# tfrecord_dir = '1_ImageRecog/data/tamucc/full_2class/'+str(TARGET_SIZE)
# # os.mkdir(tfrecord_dir)

imdir = '/media/marda/TWOTB/USGS/DATA/tamucc_coastal_imagery/subset'
csvfile = '/media/marda/TWOTB/USGS/DATA/tamucc_coastal_imagery/tamucc_subset.csv'
recoded_dir = '/media/marda/TWOTB/USGS/DATA/tamucc_coastal_imagery/subset_recoded_2class'
# os.mkdir(recoded_dir)
tfrecord_dir = '1_ImageRecog/data/tamucc/subset_2class/'+str(TARGET_SIZE)
# os.mkdir(tfrecord_dir)

dat = pd.read_csv(csvfile)


# for f,c in zip(dat.file.values, dat['class'].values):
#   if c.encode() in undev_classes:
#      shutil.copy(imdir+os.sep+f, recoded_dir+os.sep+'undev'+'_'+f)
#   else:
#      shutil.copy(imdir+os.sep+f, recoded_dir+os.sep+'dev'+'_'+f)


im, lab = read_image_and_label(recoded_dir+os.sep+'dev_IMG_8127_SecMO_Sum12_Pt3.jpg')
print(lab)

im, lab = read_image_and_label(recoded_dir+os.sep+'undev_IMG_9742_SecQN_Sum12_Pt3.jpg')
print(lab)

CLASSES = [b'dev', b'undev']

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
