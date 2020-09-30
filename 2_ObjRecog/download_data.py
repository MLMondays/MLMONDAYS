
import os, zipfile
import tensorflow as tf


# """
# ## Downloading the COCO2017 dataset
# Training on the entire COCO2017 dataset which has around 118k images takes a
# lot of time, hence we will be using a smaller subset of ~500 images for
# training in this example.
# """

url = "https://ml-mondays-data.s3-us-west-2.amazonaws.com/mlmondays_data_imrecog/releases/download/0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
tf.keras.utils.get_file(filename, url)


with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")



folder = './data'

file = 'secoora.zip'

url = "https://ml-mondays-data.s3-us-west-2.amazonaws.com/mlmondays_data_imrecog/releases/download/0.1.0/"+file
filename = os.path.join(os.getcwd(), file)
print("Downloading %s ... " % (filename))
tf.keras.utils.get_file(filename, url)
print("Unzipping to %s ... " % (folder))
with zipfile.ZipFile(file, "r") as z_fp:
    z_fp.extractall("./"+folder)

try:
    os.remove(file)
except:
    pass


#========= weights


### === scratch
file = 'secoora_retinanet_scratch_weights.zip'

folder = './retinanet'
try:
    os.mkdir(folder)
except:
    pass

folder = './retinanet/scratch'
try:
    os.mkdir(folder)
except:
    pass

url = "https://ml-mondays-data.s3-us-west-2.amazonaws.com/mlmondays_data_imrecog/releases/download/0.1.0/"+file
filename = os.path.join(os.getcwd(), file)
print("Downloading %s ... " % (filename))
tf.keras.utils.get_file(filename, url)
print("Unzipping to %s ... " % (folder))
with zipfile.ZipFile(file, "r") as z_fp:
    z_fp.extractall("./"+folder)

try:
    os.remove(file)
except:
    pass



## fine-tuned
file = 'secoora_retinanet_coco_finetune_weights.zip'

folder = './retinanet'
try:
    os.mkdir(folder)
except:
    pass

folder = './retinanet/finetune'
try:
    os.mkdir(folder)
except:
    pass


url = "https://ml-mondays-data.s3-us-west-2.amazonaws.com/mlmondays_data_imrecog/releases/download/0.1.0/"+file
filename = os.path.join(os.getcwd(), file)
print("Downloading %s ... " % (filename))
tf.keras.utils.get_file(filename, url)
print("Unzipping to %s ... " % (folder))
with zipfile.ZipFile(file, "r") as z_fp:
    z_fp.extractall("./"+folder)

try:
    os.remove(file)
except:
    pass
