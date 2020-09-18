
###############################################################
## VARIABLES
###############################################################

#start with a high validation split. If model poor on train data, can decrease
VALIDATION_SPLIT = 0.5

#start small - can increase later with larger hardware
TARGET_SIZE= 400

if TARGET_SIZE==400:
   BATCH_SIZE = 6
elif TARGET_SIZE==224:
   BATCH_SIZE = 16

num_classes = 12 #12 # 4 #2

ims_per_shard = 200

patience = 10
num_embed_dim = 8
max_epochs = 100
lr = 1e-4
