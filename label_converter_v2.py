import os
import cv2
import csv

def submit(l_path, test_path):
    label_path = l_path
    label_files = list(sorted(os.listdir(test_path)))
#     print(label_files)
    
    csvfile = open('submission.csv', 'a')
    writer = csv.writer(csvfile)
    writer.writerow(['image_id', 'PredictionString'])

    for file in label_files:
        try:
            f = open(label_path+file.replace('.jpg', '.txt'),'r')
            contents = f.readlines()
            
            # print(test_path+file)
            im = cv2.imread(test_path+file)
            h, w, c = im.shape
            # print(h, w, c)
        
            PredictionString = ''
            for content in contents:
                content = content.replace('\n','')
                c = content.split(' ')
                # print(c)

                w_center = w*float(c[1])
                h_center = h*float(c[2])
                width = w*float(c[3])
                height = h*float(c[4])
                left = int(w_center - width/2)
                # right = int(w_center + width/2)
                top = int(h_center - height/2)
                # bottom = int(h_center + height/2)
                score = float(c[5])
        
                s = str(score)+' '+str(left)+' '+str(top)+' '+str(round(width))+' '+str(round(height))+' '
                if left+round(width)-1>w-1:
                    print(file,left+round(width)-1)
                if top+round(height)-1>h-1:
                    print(file,top+round(height)-1)


                PredictionString += s
                # print(score, left, top, round(width), round(height))
            # print(PredictionString)
        
        except FileNotFoundError as e:
            print(e)
            PredictionString = str(1e-6)+' '+str(1)+' '+str(1)+' '+str(1)+' '+str(1)+' '            
            
        imgId = file.replace('.jpg', '')
        writer.writerow([imgId, PredictionString])
        


label_path = './runs/detect/exp/labels/'
testImg_path = './data/wheat/images/'
submit(label_path, testImg_path)