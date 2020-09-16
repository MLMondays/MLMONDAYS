
#===============
##2-class**
#===============

## tamucc subset 2 classes
## custom model
python tamucc_imrecog_part1a.py

## tamucc subset 2 classes
## custom model, class weights
python tamucc_imrecog_part1b.py

## tamucc full 4 classes
## cuustom model
# python tamucc_imrecog_part1c.py

#===============
##3-class
#===============

## tamucc full 3 classes
## custom model
python tamucc_imrecog_part2a.py

## tamucc subset 3 classes
## mobilenetv2 transfer learning
python tamucc_imrecog_part2b.py

## tamucc subset 3 classes
## mobilenetv2 transfer learning, with fine-tuning
python tamucc_imrecog_part2c.py

#===============
##4-class**
#===============
## tamucc subset 4 classes
## mobilenetv2 transfer learning
python tamucc_imrecog_part3a.py

# tamucc subset 4 classes
## mobilenetv2 transfer learning, class weights
python tamucc_imrecog_part3b.py

## tamucc full 4 classes
## mobilenetv2 transfer learning, class weights
python tamucc_imrecog_part3c.py


##r emember to uncomment "from nwpu_imports import *"" before you uncomment
## and run the code BELOW
# python nwpu_imrecog_part1a.py


#** = included in "live" portion of ML Mondays
