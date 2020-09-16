
###############################################################
## VARIABLES
###############################################################

#start with a high validation split. If model poor on train data, can decrease
VALIDATION_SPLIT = 0.6

#start small - can increase later with larger hardware
TARGET_SIZE= 224

if TARGET_SIZE==400:
   BATCH_SIZE = 6
elif TARGET_SIZE==224:
   BATCH_SIZE = 16

MAX_EPOCHS = 100

ims_per_shard = 200

start_lr = 1e-5 #0.00001
min_lr = start_lr
max_lr = 1e-3
rampup_epochs = 5
sustain_epochs = 0
exp_decay = .9
