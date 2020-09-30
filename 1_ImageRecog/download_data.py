
import os, zipfile
import tensorflow as tf

# os.mkdir('data')
os.mkdir('data/tamucc')


folders_to_extract_to = [
'./data',
'./data/tamucc',
'./data/tamucc',
'./data/tamucc',
'./data/tamucc',
'./data/tamucc',
]

files_to_download = [
'nwpu.zip',
'tamucc_full_2class.zip',
'tamucc_full_4class.zip',
'tamucc_subset_2class.zip',
'tamucc_subset_3class.zip',
'tamucc_subset_4class.zip',
]


for k in range(len(files_to_download)):
    file = files_to_download[k]
    folder = folders_to_extract_to[k]
    url = "https://ml-mondays-data.s3-us-west-2.amazonaws.com/mlmondays_data_imrecog/releases/download/0.1.0/"+file
    filename = os.path.join(os.getcwd(), file)
    print("Downloading %s ... " % (filename))
    tf.keras.utils.get_file(filename, url)
    print("Unzipping to %s ... " % (folder))
    with zipfile.ZipFile(file, "r") as z_fp:
        z_fp.extractall("./"+folder)


for f in files_to_download:
    try:
        os.remove(f)
    except:
        pass
