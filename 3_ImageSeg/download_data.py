
import os, zipfile
import tensorflow as tf


try:
    os.mkdir('data')
except:
    pass

folder = './data'
file = 'oysternet.zip'

url = "https://ml-mondays-data.s3-us-west-2.amazonaws.com/mlmondays_data_imrecog/releases/download/0.1.0/"+file
# url = "https://github.com/dbuscombe-usgs/mlmondays_data_imseg/releases/download/0.1.0/"+file
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


file = 'obx.zip'

url = "https://ml-mondays-data.s3-us-west-2.amazonaws.com/mlmondays_data_imrecog/releases/download/0.1.0/"+file
# url = "https://github.com/dbuscombe-usgs/mlmondays_data_imseg/releases/download/0.1.1/"+file
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
