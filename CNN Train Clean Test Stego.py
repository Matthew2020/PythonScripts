# B00092951 - Matthew Reilly
# Technological University Dublin 2019
# CNN Train Clean images & Test Stego-images

## PART 1
# Import Libraries For Classifier Model
# Libraries can CREATE NEW LAYERS & MODELS  
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

## CREATE CLASSIFIER MODEL
# Initialise the CNN
# Sequential is the easiest way to build  a CNN MODEL in Keras
CNN_Model = Sequential()

#Step 1 - Convolution
# Extract features (feature map), 
# Apply 1st Layer Filter (An image feature detector ) 
# And learn image feautre matrix and Activation applies feature map  
CNN_Model.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))

#Step 2 - Pooling
# The Pooling code will reduce the dimensions of each feature map
# but keep the most important data
# Max Pooling will take the largest element from each reformed feature map. 
CNN_Model.add(MaxPooling2D(pool_size = (2, 2)))

#Step 3 - Flattening 
#The Flattening code will convert the matrix into a linear array,   (list of finite numbers of elements stored in the memory) 
#so that it can be input into the nodes of the neural network
CNN_Model.add(Flatten())

## Step 4 - Full connection
# The Fully Connecting code will connect the convolutional network to the neural network 
CNN_Model.add(Dense(output_dim = 128, activation = 'relu'))   
CNN_Model.add(Dense(output_dim = 1, activation = 'softmax'))

## COMPILE THE CLASSIFIER MODEL
# Compiling the CNN
CNN_Model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Data augmentation to increase amunt of training data
## PART 2 - FITTING THE CNN TO THE IMAGES
from keras.preprocessing.image import ImageDataGenerator

## Configuring Training Images Settings
#  The code also resizes the batch size of the datasets as an entire dataset can’t pass into the neural network at once and fits the image to the CNN. 
rescaleTrainingDataset = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

## Configuring Testing Images Settings
# example Rescale
rescaleTestingDataset = ImageDataGenerator(rescale=1./255)

## TRAIN 1000 CLEAN IMAGE DATASET
# code assigns variables  the learning image dataset path.
# resize batch size so dataset can pass through the neural network
trainingDataset = rescaleTrainingDataset.flow_from_directory(
        'C:\\Users\\User\\Documents\\1. ITB Year 4 DFCS\\Thesis Project\\'
        'Image Recoginition Classifier Dataset\\Train Clean Test Stego Dataset\\'
        'Train 1000 Clean Test 490 Stego Dataset\\Train 1000 Clean Image Dataset',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

## TEST 490 STEGO IMAGES
# code assigns variables  the testing image dataset path.
# resize batch size so dataset can pass through the neural network
testingDataset= rescaleTestingDataset.flow_from_directory(
        'C:\\Users\\User\\Documents\\1. ITB Year 4 DFCS\\Thesis Project\\'
        'Image Recoginition Classifier Dataset\\Train Clean Test Stego Dataset\\'
        'Train 1000 Clean Test 490 Stego Dataset\\Test 490 Stego Image Dataset',
        target_size=(64, 64),       
        batch_size=32,
        class_mode='binary')

## Epoch set to 100  
# the dataset will pass through the neural network 100 times 
CNN_Model_Plot_Chart = CNN_Model.fit_generator(        
        trainingDataset,
        steps_per_epoch = 8,
        epochs=10,
        validation_data = testingDataset,
        validation_steps=8,
        verbose=1)                      


# History Loss/Accuracy class import function for Graph Design
from lossAndAccuracyPlotChart import lossAccuracyPlotChartFunction

# Call function for Graphs in  historyLossAccuracyFunction class
lossAccuracyPlotChartFunction(CNN_Model_Plot_Chart)





















## CLASSIFIER 1ST CODE
## TRAINING The Model
#from IPython.display import display 
#from PIL import Image

#classifierModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#classifierModel.fit_generator(
#        training_set,
#        steps_per_epoch = 1000,
#        epochs=5,
#        validation_data = test_set,
#        validation_steps=100,
#        verbose=1)
# CLASSIFIER 1ST CODE



