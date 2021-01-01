import os
import cv2
import numpy as np
abs_path = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(abs_path, 'original', 'images')
label_dir = os.path.join(abs_path, 'original', 'tsv')
crop_img_dir = os.path.join(abs_path, 'crop', 'images')
crop_label_dir = os.path.join(abs_path, 'crop', 'labels')
 

def shift_bbox(bboxes, value):
    value = np.array(value)
    return bboxes - value

def cut_bbox(bboxes, mode, shrink=True, prune_small=True):
    cut = {"r": lambda arr: np.delete(arr, np.unique(np.where(arr[:, 0] >= 1024)), axis=0),
           "b": lambda arr: np.delete(arr, np.unique(np.where(arr[:, 1] >= 1024)), axis=0),
           "l": lambda arr: np.delete(arr, np.unique(np.where(arr[:, 2] <  0   )), axis=0),
           "t": lambda arr: np.delete(arr, np.unique(np.where(arr[:, 3] <  0   )), axis=0)}
    mode = mode.lower()
    for s in mode:
        bboxes = cut[s](bboxes)

    if shrink:
        bboxes = np.clip(bboxes, 0, 1023)

    # if prune_small:
    #     bboxes_wh = bboxes[:,2:] - bboxes[:,:2]
    #     bboxes = np.delete(bboxes, np.unique(np.where(bboxes_wh<35)[0]), axis=0)
    
    return bboxes

def crop_image(visualize=False, cat_crop=False, save=False):
    img_names = os.listdir(img_dir)
    bad = []
    for img_name in img_names:
        print(img_name)
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)/255
        h, w, c = img.shape
        crop_imgs = {
            0 : img[:1024,  :1024,  :].copy(),# left top
            1 : img[:1024,  -1024:, :].copy(),# right top
            2 : img[-1024:, :1024,  :].copy(),# left bottom
            3 : img[-1024:, -1024:, :].copy() # right bottom
        }

        lable_path = os.path.join(label_dir, img_name.replace('jpg', 'bboxes.tsv'))
        label = np.loadtxt(lable_path, dtype=np.int)      
        crop_bboxes = {0: cut_bbox(shift_bbox(label.copy(), [0     , 0     ]*2), 'rb'),
                       1: cut_bbox(shift_bbox(label.copy(), [w-1024, 0     ]*2), 'lb'),
                       2: cut_bbox(shift_bbox(label.copy(), [0     , h-1024]*2), 'tr'),
                       3: cut_bbox(shift_bbox(label.copy(), [w-1024, h-1024]*2), 'tl')}
        
        for k in crop_bboxes:
            if len(crop_bboxes[k])<=25:
                bad.append(img_name.replace('.jpg', '_crop{}.jpg'.format(k)))

        if visualize:
            if cat_crop:            
                boxed = np.zeros((h, w, c))
                img_cp = crop_imgs[0].copy()
                for bbox in crop_bboxes[0]:
                    img_cp = cv2.rectangle(img_cp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                boxed[:1024,  :1024,  :] += img_cp*0.5
                img_cp = crop_imgs[1].copy()
                for bbox in crop_bboxes[1]:
                    img_cp = cv2.rectangle(img_cp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                boxed[:1024,  -1024:, :] += img_cp*0.5
                img_cp = crop_imgs[2].copy()
                for bbox in crop_bboxes[2]:
                    img_cp = cv2.rectangle(img_cp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                boxed[-1024:, :1024,  :] += img_cp*0.5
                img_cp = crop_imgs[3].copy()
                for bbox in crop_bboxes[3]:
                    img_cp = cv2.rectangle(img_cp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                boxed[-1024:, -1024:, :] += img_cp*0.5
                cv2.imshow(img_name, cv2.resize(boxed, (w//2,h//2)))

            else:
                for k in crop_imgs:
                    img_cp = crop_imgs[k].copy()
                    for bbox in crop_bboxes[k]:
                        img_cp = cv2.rectangle(img_cp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                    # if len(crop_bboxes[k])<=30:
                    #     cv2.putText(img_cp, "bad example", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(str(k), cv2.resize(img_cp,(480,480)))
                cv2.moveWindow('0', 0,   0  )
                cv2.moveWindow('1', 640, 0  )
                cv2.moveWindow('2', 0,   640)
                cv2.moveWindow('3', 640, 640)

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break
        
        if save:
            if "_130218" in img_name:
                img_name = img_name.replace("_130218", '')
            for k in crop_imgs:
                crop_img_name = img_name.replace('.jpg', '_crop{}.jpg'.format(k))
                crop_img_path = os.path.join(crop_img_dir, crop_img_name)
                cv2.imwrite(crop_img_path, crop_imgs[k])
            
                h_c, w_c, _ = crop_imgs[k].shape
                # write label
                bboxes =crop_bboxes[k]
                left_top = bboxes[:,:2]
                right_bottom = bboxes[:,2:]
                center = (left_top + right_bottom) / 2
                size = right_bottom - left_top 
                bboxes = np.hstack((center, size)) / np.array([[w_c, h_c, w_c, h_c]])# normalize [c_x, c_y, w, h]
                bboxes = bboxes.round(6)
                assert(np.max(bboxes) <= 1)# check normalization

                cate_id = np.zeros((bboxes.shape[0], 1))
                labels = np.hstack((cate_id, bboxes))

                label_path = crop_img_path.replace('jpg', 'txt').replace('images', 'labels')
                np.savetxt(label_path, labels, fmt='%d %.6f %.6f %.6f %.6f')
        
    # for pth in bad:
    #     if "_130218" in pth:
    #         pth = pth.replace("_130218", '')
    #     img_pth = os.path.join(crop_img_dir,pth)
    #     os.rename(img_pth, img_pth.replace('images','bad'))
    #     label_pth = img_pth.replace('jpg','txt').replace('images','labels')
    #     os.rename(label_pth, label_pth.replace('labels','bad'))
        
if __name__ == "__main__":
    crop_image(True,True,False)
