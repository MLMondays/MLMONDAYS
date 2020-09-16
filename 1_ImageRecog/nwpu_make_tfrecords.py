
# code to standardize the naming of the files (removing underscores, of variable number)
# for file in *; do
# echo "${file/_/}"
# mv "$file" "${file/_/}"
# done
#do twice for sea ice

from imports import *
print(TARGET_SIZE)

imdir = '/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/1_ImageRecog/data/nwpu/images'
tfrecord_dir = '/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/1_ImageRecog/data/nwpu/full/'+str(TARGET_SIZE)

images=tf.io.gfile.glob(imdir+os.sep+'*.jpg')

# get file names
labels = [i.split('/')[-1] for i in images]

# remove numbers
labels = [''.join([i for i in s if not i.isdigit()]) for s in labels]

# remove file extension
labels = [i.split('.jpg')[0] for i in labels]


CLASSES = np.unique(np.array(labels))
## 11 classes

print(CLASSES)

# need a different function because the file structure is different than that of the tamucc imagery

def read_image_and_label(img_path):

  bits = tf.io.read_file(img_path)
  image = tf.image.decode_jpeg(bits)

  label = tf.strings.split(img_path, sep='/')

  # remove numbers
  label = tf.strings.regex_replace(label[-1], "([0-9]+)", r"")
  ##label = tf.strings.split(''.join([i for i in label[-1].numpy().decode() if not i.isdigit()]), sep='.jpg')
  label = tf.strings.split(label, sep='.jpg')

  return image,label[0]


# overwrite get_dataset_for_tfrecords (from imports.py) to incorporate the redefined read_image_and_label
def get_dataset_for_tfrecords(recoded_dir, shared_size):
    tamucc_dataset = tf.data.Dataset.list_files(recoded_dir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
    tamucc_dataset = tamucc_dataset.map(read_image_and_label)
    tamucc_dataset = tamucc_dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)

    tamucc_dataset = tamucc_dataset.map(recompress_image, num_parallel_calls=AUTO)
    tamucc_dataset = tamucc_dataset.batch(shared_size)
    return tamucc_dataset


for c in CLASSES:
    im, lab = read_image_and_label(imdir+os.sep+c+'100.jpg')
    print(lab.numpy())
    class_num = np.argmax(np.array(CLASSES)==lab)
    print(class_num)

CLASSES = [c.encode() for c in CLASSES]

nb_images=len(tf.io.gfile.glob(imdir+os.sep+'*.jpg'))

SHARDS = int(nb_images / ims_per_shard) + (1 if nb_images % ims_per_shard != 0 else 0)
print(SHARDS)

shared_size = int(np.ceil(1.0 * nb_images / SHARDS))
print(shared_size)

# nwpu_dataset = get_dataset_for_tfrecords(imdir, BATCH_SIZE)
#
# for imgs,lbls in nwpu_dataset.take(1):
#   #print(lbls)
#   for count,im in enumerate(imgs):
#      plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
#      plt.imshow(tf.image.decode_jpeg(im, channels=3))
#      plt.title(lbls.numpy()[count].decode(), fontsize=8)
#      #plt.axis('off')
# plt.show()

nwpu_dataset = get_dataset_for_tfrecords(imdir, shared_size)


write_records(nwpu_dataset, tfrecord_dir, CLASSES)
