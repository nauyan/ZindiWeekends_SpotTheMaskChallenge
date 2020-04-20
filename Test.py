import keras
import glob
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import os 
import pandas as pd
import json

# Loading Config File
with open('./config.json') as config_file:
    config = json.load(config_file)


im_width = config['im_width']
im_height = config['im_height']
testList = glob.glob(config['TestFolder']+"/*")

testList[:] = [os.path.basename(x) for x in testList] 
testSet_Name = []
testSet_IMG = []
testSet_LABEL = []
for x in testList:
    name = glob.glob(config['TestFolder']+"/"+x)
    img = img_to_array(load_img(name[0], color_mode='rgb', target_size=[im_width,im_height]))
    testSet_Name.append(x)
    testSet_IMG.append(img)
    testSet_LABEL.append(np.nan)  

testSet_Name = np.asarray(testSet_Name)
testSet_IMG = np.asarray(testSet_IMG)
testSet_LABEL = np.asarray(testSet_LABEL) 
testSet_IMG = testSet_IMG/255.0

import efficientnet.keras

model = keras.models.load_model("Best.h5") 

testSet_LABEL = model.predict(testSet_IMG)
#print(testSet_LABEL)

labels_updated = np.zeros(testSet_LABEL.shape[0],float)
count = 0
for x in testSet_LABEL:
    labels_updated[count] = x
    count = count + 1

df = pd.DataFrame()
df['image']  = testSet_Name
df['target']  = labels_updated
print(df)
df.to_csv('submission.csv', index=False)