import glob
import cv2
import pandas as pd 
import numpy as np
import shutil  
import os
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical

from zipfile import ZipFile
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import efficientnet.keras as efn 
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D

import shutil
import json

# Loadig Config File
with open('./config.json') as config_file:
    config = json.load(config_file)

# Removing and Cleaning the Repo
compressFolder = config['compressFolder']
try:
    shutil.rmtree(compressFolder)
except OSError as e:
    print("Error: %s : %s" % (compressFolder, e.strerror))

TestFolder = config['TestFolder']
try:
    shutil.rmtree(TestFolder)
except OSError as e:
    print("Error: %s : %s" % (TestFolder, e.strerror))
    
TrainFolder = config['TrainFolder']
try:
    shutil.rmtree(TrainFolder)
except OSError as e:
    print("Error: %s : %s" % (TrainFolder, e.strerror))

compressFile = config['compressFile']
with ZipFile(compressFile, 'r') as zipObj:
   zipObj.extractall()
 

# Making Test and Train Directories 
os.mkdir(TestFolder);
os.mkdir(TrainFolder);


# Reading Label File and Storing to Pandas Dataframe
data = pd.read_csv(config['LabelFiles'])
trainList = data.values

# Moving Train Data to Respective Folder
for x in trainList:
    source = "./"+str(compressFolder)+"/"+str(x[0])
    destination = "./"+str(TrainFolder)+"/"+str(x[0])
    dest = shutil.move(source, destination)  
testList = glob.glob('./'+str(compressFolder)+'/*')

# Moving Test Data to Respective Folder
for x in testList:
    source = x
    destination = "./"+str(TestFolder)+"/"+str(x[0])
    dest = shutil.move(source, destination)
 
 
im_width = config['im_width']
im_height = config['im_height']
trainSet_NAME = []
trainSet_IMG = []
trainSet_LABEL = []
testSet_Name = []
testSet_IMG = []
testSet_LABEL = []

# Preparing Training Data
for x in trainList:
    name = glob.glob("./"+str(TrainFolder)+"/"+x[0])
    img = img_to_array(load_img(name[0], color_mode='rgb', target_size=[im_width,im_height])) 
    trainSet_NAME.append(x[0])
    trainSet_IMG.append(img)
    trainSet_LABEL.append(x[1])

# Preparing Test Data    
testList[:] = [os.path.splitext(os.path.basename(x))[0] for x in testList]    
for x in testList:
    name = glob.glob("./TestImages/"+x+"*")
    img = img_to_array(load_img(name[0], color_mode='rgb', target_size=[im_width,im_height]))
    testSet_Name.append(x)
    testSet_IMG.append(img)
    testSet_LABEL.append(np.nan)    
    
    
print("Train Data Shape "+str(len(trainSet_NAME))) 
print("Test Data Shape "+str(len(testSet_Name)))  

# Conversion of Lists into Numpy Array and Normalizing Images
trainSet_NAME = np.asarray(trainSet_NAME)
trainSet_IMG = np.asarray(trainSet_IMG)
trainSet_LABEL = np.asarray(trainSet_LABEL) 
trainSet_IMG = trainSet_IMG/255.0

# Conversion of Lists into Numpy Array and Normalizing Images
testSet_Name = np.asarray(testSet_Name)
testSet_IMG = np.asarray(testSet_IMG)
testSet_LABEL = np.asarray(testSet_LABEL) 
testSet_IMG = testSet_IMG/255.0

print("Train Data Shape "+str(trainSet_IMG.shape))
print("Test Data Shape "+str(testSet_IMG.shape))

# Test and Train Split of Data
X_train, X_test, y_train, y_test = train_test_split(trainSet_IMG, trainSet_LABEL, test_size=0.20)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# Training Model Layers Arrangment
baseModel = efn.EfficientNetB0(weights="imagenet", include_top=False,input_shape=X_train[0].shape)
headModel = baseModel.output
headModel = GlobalAveragePooling2D(name='avg_pool')(headModel)
headModel = Dropout(config['Dropout'], name='top_dropout')(headModel)
headModel = Dense(1,activation='sigmoid')(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

# Loss Function and its Parameters
adam = keras.optimizers.Adam(amsgrad=True)

# Compilation of Model
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

# Setting up of Callbacks for the Model
callbacks = [
    EarlyStopping(patience=15, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1),
    ModelCheckpoint('./Best.h5', monitor='val_loss', mode = 'min' , verbose=1, save_best_only=True, save_weights_only=False)
]

# Seting up of Data Augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	fill_mode="nearest")

# Setting up of Training Parameters and Starting Training    
results = model.fit_generator(aug.flow(X_train, y_train, batch_size=config['Batch']),
	validation_data=(X_test, y_test), epochs=config['Epochs'], callbacks=callbacks)

# Evaluting Model 
print(model.evaluate(X_test, y_test, verbose=1))

# Train and Test Validation Loss Plots
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.savefig('./train_loss.png')

# Train and Test Accuracy Loss Plots
plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["accuracy"], label="accuracy")
plt.plot(results.history["val_accuracy"], label="val_accuracy")
plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend();
plt.savefig('./train_accuracy.png')
