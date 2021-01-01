# To training this model
- prepare data
    1. Download the dataset [here](https://www.kaggle.com/c/global-wheat-detection)
    3. In `kaggle_label_cvt.py` I implemented a simple converer to conver the label to the format required.
    4. You can check the converted label at the following path 'data/wheat/global_wheat_detect'
- train
    - type `python train.py --help` to get more information
- for more detial, please refer to this [tutorial](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)



# External Dataset
- by default, I only use the data provied by the kaggle competition to train this model, but SPIKE dataset is recommended.
- You can download SPIKE dataset [here](https://sourceforge.net/projects/spike-dataset/)
    1. In `SPIKE_label_cvt.py` I implemented a simple converer to conver the label to the format required.
    2. (option) In `crop.py` I implemented the four corner crop and data clean to drop some images has poor label quality, but it did not work very well.

# Make a submission
Because global-wheat-competition is a code competition we must upload our code and run it on the remote server. However, the GPU time is limited. Only 30 hours is available every week. Thus, we train our model on the local machine; then, upload our pre-train model weight as the dataset and import it in the Kaggle notebook.

Here is the [linl](https://www.kaggle.com/chia56028/yolov5-stable?fbclid=IwAR3X1W-hvvBHh9MbDbOOaRN5vDdVHGKOKB0EKUF_8MU1FyA08ksovyT3v_g) to our notebook. In this notebook, we also do some prediction time data augmentation such as Pseudo Label, and TTA. Please refer to the notebook to get more information.
