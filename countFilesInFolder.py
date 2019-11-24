# B00092951 - Matthew Reilly
# Technological University Dublin 2019
# Count how many files are in a folder

# Import OS Module
import os

# Path to folder with image files
locationPathFolder = 'C:\\Users\\User\\Documents\\1. ITB Year 4 DFCS\\Thesis Project\\Image Recoginition Classifier Dataset\\Train Clean Test Stego Dataset\\Train 1000 Clean Test 490 Stego Dataset\\Train 1000 Clean Image Dataset\\train2'

# Assign variable 
listOfFiles = os.listdir(locationPathFolder)


# Function Defined To Print how many files are in folder
def functionToPrintFiles():
    
    # Print the number of files
    print(len(listOfFiles))
   

# Call Function
functionToPrintFiles()



