import os
import random

prefix = 'new_SPIKE_'
# global_wheat_detect
imgs = os.listdir('global_wheat_detect/images')
random.shuffle(imgs)
train_imgs = ["data/wheat/global_wheat_detect/images/" + imgs[i] for i in range(0, 3000)]
val_imgs = ["data/wheat/global_wheat_detect/images/" + imgs[i] for i in range(3000, 3422)]

# SPIKE
imgs = os.listdir('SPIKE/crop/images')
for img in imgs:
    if 'test' in img:
        val_imgs.append("data/wheat/SPIKE/crop/images/" + img) 
    else:
        train_imgs.append("data/wheat/SPIKE/crop/images/" + img) 

random.shuffle(train_imgs)
random.shuffle(val_imgs)
with open(prefix+'train.txt', 'w') as f:
    f.write('\n'.join(train_imgs))

with open(prefix+'val.txt', 'w') as f:
    f.write('\n'.join(val_imgs))