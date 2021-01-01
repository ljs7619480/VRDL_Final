import os
import cv2
import pandas as pd

output_dir = 'runs/detect/exp/labels/'
image_dir = 'data/wheat/images/'
result = {'image_id': [], 'PredictionString': []}

for img in os.listdir(image_dir):
    img_id = img[:-4]
    PredictionList = []
    try:
        with open(os.path.join(output_dir, img_id+'.txt'), 'r') as f:
            img_h, img_w, _ = cv2.imread(image_dir+img).shape
            items = f.readlines()
            
            for item in items:
                c_x, c_y, b_w, b_h, conf = [float(element) for element in item.strip().split(' ')[1:]]
                c_x = round((c_x - b_w/2) * img_w)
                c_y = round((c_y - b_h/2) * img_h)
                b_w = round(b_w * img_w)
                b_h = round(b_h * img_h)
                conf = round(conf,1)
                PredictionList.append("{} {} {} {} {}".format(conf, c_x, c_y, b_w, b_h))
            
    except FileNotFoundError as e:
        print(e)

    finally:
        result['image_id'].append(img_id)
        result['PredictionString'].append(' '.join(PredictionList))



df = pd.DataFrame(result)
df.to_csv("submission.csv",index=False)

# for img in os.listdir(output_dir):
#     with open(os.path.join(output_dir, img), 'r') as f:
#         img_id = img[:-4]
#         img_h, img_w, _ = cv2.imread(output_dir+'../'+img_id+'.jpg').shape
#         items = f.readlines()
        
#         PredictionList = []
#         for item in items:
#             c_x, c_y, b_w, b_h, conf = [float(element) for element in item.strip().split(' ')[1:]]
#             c_x = round((c_x - b_w/2) * img_w)
#             c_y = round((c_y - b_h/2) * img_h)
#             b_w = round(b_w * img_w)
#             b_h = round(b_h * img_h)
#             conf = round(conf,1)
#             PredictionList.append("{} {} {} {} {}".format(conf, c_x, c_y, b_w, b_h))
        
#         result['image_id'].append(img_id)
#         result['PredictionString'].append(' '.join(PredictionList))

# df = pd.DataFrame(result)
# df.to_csv("submission.csv",index=False)