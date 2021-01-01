import numpy as np
import os
import cv2

abs_path = os.path.dirname(os.path.abspath(__file__))
tsv_dir = os.path.join(abs_path, 'original/tsv')

for file in ['Spike_0068.bboxes.tsv']:
# for file in os.listdir(tsv_dir):
    print(file)
    tsv_path = os.path.join(tsv_dir, file)
    bboxes = np.loadtxt(tsv_path, np.float32)# [left,top,right,bottom]
    
    img_path = tsv_path.replace('bboxes.tsv','jpg').replace('tsv', 'images')
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    for bbox in bboxes:
        img = cv2.rectangle(img,(bbox[0],bbox[1]), (bbox[2],bbox[3]), (0, 255, 0), thickness=3)
    cv2.imshow(file,cv2.resize(img, (w//2, h//2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('123.jpg',img)

    # left_top = bboxes[:,:2]
    # right_bottom = bboxes[:,2:]
    # center = (left_top + right_bottom) / 2
    # size = right_bottom - left_top 
    # bboxes = np.hstack((center, size)) / np.array([[w, h, w, h]])# normalize [c_x, c_y, w, h]
    # bboxes = bboxes.round(6)
    # assert(np.max(bboxes) <= 1)# check normalization

    # cate_id = np.zeros((bboxes.shape[0], 1))
    # labels = np.hstack((cate_id, bboxes))

    # label_path = img_path.replace('jpg', 'txt').replace('images', 'labels')
    # np.savetxt(label_path, labels, fmt='%d %.6f %.6f %.6f %.6f')