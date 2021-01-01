import pandas as pd
import json

df = pd.read_csv('train.csv')

objs = {}
for index, row in df.iterrows():
    img_path = row['image_id']
    if not (img_path in objs):
        objs[img_path] = []

    cate = 0
    bbox = json.loads(row['bbox'])
    cent_x = (bbox[0]+bbox[2]/2)/row['width']
    cent_y = (bbox[1]+bbox[3]/2)/row['height']
    size_x = bbox[2]/row['width']
    size_y = bbox[3]/row['height']
    assert(cent_x<=1 and cent_y<=1 and size_x<1 and size_y<1)
    objs[img_path].append("{} {:6f} {:6f} {:6f} {:6f}".format(cate, cent_x, cent_y, size_x, size_y))

for path, meta in objs.items():
    with open('label/{}.txt'.format(path), 'w') as f:
        f.write('\n'.join(meta))


# recover
# import cv2
# import numpy as np
# img = cv2.imread('train/b6ab77fd7.jpg')
# for obj in objs['b6ab77fd7']:
#     meta = np.array([float(zz) for zz in obj.split(' ')][1:])*1024
#     cent = meta[0:2]
#     size = meta[2:4]
#     pt1 = np.round(cent-size/2)
#     pt2 = np.round(cent+size/2)
#     print(pt1[0], pt1[1], pt2[0]-pt1[0], pt2[1]-pt1[1])
