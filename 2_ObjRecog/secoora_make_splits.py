
import pandas as pd
import os, shutil
import random

dat = pd.read_csv('data/secoora/labels.csv')

dat = dat.sample(frac=1).reset_index(drop=True)
print(dat)

files = dat.filename.values
# random.shuffle(files)

train_files = files[:5115]
val_files = files[5115:]

for k in train_files:
    shutil.copy(os.getcwd()+'/data/secoora/images/'+k, os.getcwd()+'/data/secoora/train/')


for k in val_files:
    shutil.copy(os.getcwd()+'/data/secoora/images/'+k, os.getcwd()+'/data/secoora/val/')


# files = sorted(dat.filename.values)

res = []
for i, k in enumerate(train_files):
    res.append(files.tolist().index(k))

train_dat = dat.iloc[res]

train_dat.to_csv('train.csv')


res = []
for i, k in enumerate(val_files):
    res.append(files.tolist().index(k))

val_dat = dat.iloc[res]

val_dat.to_csv('val.csv')
