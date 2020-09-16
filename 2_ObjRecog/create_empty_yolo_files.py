from glob import glob
import os

for cond in ['test', 'train','validation']:

    jpg = glob(cond+'/*.jpg')

    for f in jpg:
       file_query = f.replace('jpg','txt').replace(cond, cond+'_labels')
       if os.path.isfile(file_query):
          pass
       else:
          print("Creating %s" % (file_query))
          with open(file_query, 'w') as fp:
             pass
