# -*- coding: utf-8 -*-
"""
@author: Matthew2014
"""

## PART 1
# Import Libraries For Classifier Model
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense




## CREATE CLASSIFIER MODEL
#Initialise the CNN
classifierModel = Sequential()

#Step 1 - Convolution 
classifierModel.add(Convolution2D(64, 7, 7, input_shape = (64,64,3), activation = 'relu'))

#Step 2 - Pooling
classifierModel.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening 
classifierModel.add(Flatten())

#Step 4 - Full connection
classifierModel.add(Dense(output_dim = 128, activation = 'relu'))   
classifierModel.add(Dense(output_dim = 1, activation = 'softmax'))

## COMPILE THE CLASSIFIER MODEL
#Compiling the CNN
classifierModel.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


## Data Augmentation
##PART 2 - FITTING THE CNN TO THE IMAGES
from keras.preprocessing.image import ImageDataGenerator

##Configuring Training Images Settings
# example Rescale
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

##Configuring Testing Images Settings
# example Rescale
test_datagen = ImageDataGenerator(rescale=1./255)

##TRAIN 1000 Mixed CLEAN & Stego IMAGE DATASET
trainingDataset = train_datagen.flow_from_directory(
        'C:\\Users\\User\\Documents\\1. ITB Year 4 DFCS\\Thesis Project\\info\\Train Test Mixed Clean & Stego\\Train 1000 Mixed Clean & Stego Dataset',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

## TEST 490 Mixed CLEAN & STEGO IMAGES
testingDataset= test_datagen.flow_from_directory(
        'C:\\Users\\User\\Documents\\1. ITB Year 4 DFCS\\Thesis Project\\info\\Train Test Mixed Clean & Stego\\Test 490 Mixed Clean & Stego Dataset',
        target_size=(64, 64),       
        batch_size=32,
        class_mode='binary')

##Run Epochs Through Training and Testing Dataset
classifierModelHistroy = classifierModel.fit_generator(        
        trainingDataset,
        steps_per_epoch = 80,
        epochs=100,
        validation_data = testingDataset,
        validation_steps=8,
        verbose=1)                      


# History Loss/Accuracy class import function for Graph Design
from historyLossAccuracyClass import historyLossAccuracyFunction

##Call function for Graphs in  historyLossAccuracyFunction class
historyLossAccuracyFunction(classifierModelHistroy)
