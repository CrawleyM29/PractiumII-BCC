# PractiumII-Breast Cancer Classification 

## Abstract

Creating an AI that will classify breast cancer as ether Benign or Malignant using Python language (Spyder Program with Anaconda). Using Deep Learning to create an AI (Artificial Intelligence) that will build a classifier to train 80% of breast cancer histology image dataset, keeping 10% of the data for validation purposes.  Keras will be used and help define a Convolutional Neural Network (CNN) and naming it 'CancerNet', training the images.  A confusion matrix will help analyze the performance of the model.

Invasive Ductal Carcinoma (IDC) is a cancer that develops in the milk duct, invading the fibrous and/or fatty breast tissue outside the duct.  This type of cancer is most common type of brease cancer forming 80% of all breast cacer diagnoses.  Histology is the study of the microscopic structure of tissues.

## About the Dataset

I am uzing the IDC_regular dataset (histology image dataset) from Kaggle.  The dataset holds 2,077,524 patches that are 50x50 in size and extracted from 162 whole mount slide images of breast cancer specimens that were scanned at 40x.  In these images, 1,098,738 test negative 78,786 are tested positive with IDC. This dataset is available in public domain and you can be downloaded [here](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).

You will need to install some python packages to be able to run this advanced python project.  You can do this with pip-
        
        pip install numpy opencv-python pillow tensorflow keras imutils scikit-learn matplotlib
      
Inside the inner breast-cancer-classification directory, I created a directory datasets- inside this, create directory original

        mkdir datasets
        mkdir datasets\original 

## Config.py

Config.py holds some configurations we will need for building the dataset and training the model.  You'll find this in the directory 'cancernet'.  Due to the nature of my project, I don't have models until the end of training.

       import os

       INPUT_DATASET = "datasets/original"

       BASE_PATH = "datasets/idc"
       TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
       VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
       TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

       TRAIN_SPLIT = 0.8
       VAL_SPLIT = 0.1

I declared the path to the input dataset (.../datasets/original), the new directory (.../datasets/idc), and the validation, training, and testing path directories using the base path.  Also, declaring that 80% of our dataset will be used for training purposes while 10% of the training data is used for validation.

## build_dataset.py

This section will be splitting our dataset into testing sets, training, and validation in the mentioned ratio above - 80% for training (of that, 10% for validation) and 20% for testing.  Using Keras, ImageDataGenerator, batches of images will be extracted to help avoid making space for all of our dataset at once.

      from cancernet import config
      from imutils import paths
      import random, shutil, os

      originalPaths=list(paths.list_images(config.INPUT_DATASET))
      random.seed(7)
      random.shuffle(originalPaths)

      index=int(len(originalPaths)*config.TRAIN_SPLIT)
      trainPaths=originalPaths[:index]
      testPaths=originalPaths[index:]

      index=int(len(trainPaths)*config.VAL_SPLIT)
      valPaths=trainPaths[:index]
      trainPaths=trainPaths[index:]

      datasets=[("training", trainPaths, config.TRAIN_PATH),
          ("validation", valPaths, config.VAL_PATH),
          ("testing", testPaths, config.TEST_PATH)]

      for (setType, originalPaths, basePath) in datasets:
          print(f'Building {setType} set')

          if not os.path.exists(basePath):
                print(f'Building directory {base_path}')
                os.makedirs(basePath)

          for path in originalPaths:
                file=path.split(os.path.sep)[-1]
                label=file[-5:-4]

                labelPath=os.path.sep.join([basePath,label])
                if not os.path.exists(labelPath):
                        print(f'Building directory {labelPath}')
                        os.makedirs(labelPath)

                newPath=os.path.sep.join([labelPath, file])
                shutil.copy2(inputPath, newPath)

Original paths will be built with the images, importing os, shutil, imutils, random, and config, and shuffle the list (our images).  Next, we calculate an index by multiplying the list length by 0.8 to help slice this list to get sublists for training and testing.  We need to also calculate the index to save 10% for training the dataset for validation and keeping the rest for training itself.

Tuples is used for information about the training, testing sets, and validation--holding the paths and base path for each. Each setType, base path, and path in the list, we'll print, saying 'Building testing set', while also creating a directory if the path does not exist.  

![Building Training Set](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/Building-Training-Set.JPG)

   
## Cancernet.py

The CNN (Convolutional Neural Network) will be built and nameing the network 'CancerNet'.  The network will perform the following:

        -Using 3,3 CONV filter(s)
        -Perform max-pooling
        -Stacking filters on top of one another
        -Using depthwise separable convolutions for more efficincy and consume less memory
        
![Building Cancernet](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/CancerNet.JPG)

Sequential API is being used to help build CancerNet and SeparableConv2D for depthwise convolutions.  CancerNet has a static method build with four parameters: height and width of the images, the depth (amount of color channels in each image), and the number of classes the network will predict between (for us there are 2: 0 and 1).  

## Train_model.py

 ### Output 1st Half
       
![Cancernet 1st Half](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/cancernet1.JPG)

### Output 2nd Half

![Cancernet 2nd Half](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/cancernet2.JPG)
