# Practium II-Breast Cancer Classification 

## Abstract

Creating an AI that will classify breast cancer as ether Benign or Malignant using Python language (Spyder Program with Anaconda). Using Deep Learning to create an AI (Artificial Intelligence) that will build a classifier to train 80% of breast cancer histology image dataset, keeping 10% of the data for validation purposes.  Keras will be used and help define a Convolutional Neural Network (CNN) and naming it 'CancerNet', training the images.  A confusion matrix will help analyze the performance of the model.

Invasive Ductal Carcinoma (IDC) is a cancer that develops in the milk duct, invading the fibrous and/or fatty breast tissue outside the duct.  This type of cancer is most common type of brease cancer forming 80% of all breast cacer diagnoses.  Histology is the study of the microscopic structure of tissues.

### Notes

My slides are only showcasing my Deep Learning as that is what my class is for.  However, I am including Data Exploraton (before AI Training) to showcase skills for future opportunities.

## Table of Contents

  1. About the Dataset
  2. Config.py
  3. Build_dataset.py
  4. Cancernet.py
  5. Train_model.py
  6. Deep Learning Results
  7. Data Exploration Results
  8. Summary
  9. References
        
## 1. About the Dataset

I am uzing the IDC_regular dataset (histology image dataset) from Kaggle.  The dataset holds 2,077,524 patches that are 50x50 in size and extracted from 162 whole mount slide images of breast cancer specimens that were scanned at 40x.  In these images, 1,098,738 test negative 78,786 are tested positive with IDC. This dataset is available in public domain and you can be downloaded [Here](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

You will need to install some python packages to be able to run this advanced python project.  You can do this with pip-
        
        pip install numpy opencv-python pillow tensorflow keras imutils scikit-learn matplotlib
      
Inside the inner breast-cancer-classification directory, I created a directory datasets- inside this, create directory original

        mkdir datasets
        mkdir datasets\original 


## 2. Config.py

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

## 3. Build_dataset.py

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

   
## 4. Cancernet.py

The CNN (Convolutional Neural Network) will be built and nameing the network 'CancerNet'.  The network will perform the following:

        -Using 3,3 CONV filter(s)
        -Perform max-pooling
        -Stacking filters on top of one another
        -Using depthwise separable convolutions for more efficincy and consume less memory
        
![Building Cancernet](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/CancerNet.JPG)

Sequential API is being used to help build CancerNet and SeparableConv2D for depthwise convolutions.  CancerNet has a static method build with four parameters: height and width of the images, the depth (amount of color channels in each image), and the number of classes the network will predict between (for us there are 2: 0 and 1).  

## 5. Train_model.py

We are now train and evaluate our model by importing from keras, cancernet, sklearn, config, imutils, matplotlib, and os.

By setting initial values for the number of epochs, the learning rate, and batch size, we will get the number of paths in the three directories for training, testing, and validation.  After, we will then get the class weight for the training data so we can deal with the imbalance.  We then initilize the training data augmentation object-- a process of regularization which helps generalize the model.  At this time, we slightly modify the training examples to help avoid the need for more training data.

 ### Output 1st Half
       
![Cancernet 1st Half](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/cancernet1.JPG)

### Output 2nd Half

![Cancernet 2nd Half](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/cancernet2.JPG)

We have successfully trained our dataset!  Results are below.

## 6. Results for Deep Learning

I am seperating my results into two sections: Results in Data Training using Deep Learning, and Data Exploration to learn our data.

Deep Learning and training my AI is the main focus for my project.  However, I wanted to see what my data was telling me story wise for future insights purposes.

### Results 1: Dropping ID and Unnamed

I removed patient ID's and those that are not named malignant or benign.  We now have 2-columns that show the total that are left over:

![Results1](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/Not.trained_MALvsBEN.JPG)

### Results 2: Hyper Parameter Tuning

A goal of mine was to reduce the False Negatives (FN) due to tumors  which are malignant should not be classified as benign even if that means the model may classify a few benign tumors as malgnant.  By using sklearn's 'fbeta_score' as our scoring function with GridSearchCV.A beat > 1 to make 'fbeta_score' favor our recall over precision.

At first, our grid searching of 'M' is a 2 score which means that our FN's are showing as Malgnant.  We really need to get this a 1:

![FNScore2](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/False_Neg%202.score.JPG)

After setting the decision threshold to 0.42, we have succuffully reached our goal of FN to 1:

![FNScore1](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/FT_Score.1.JPG)

### Results 3: Deep Learning - Training

Deep Learning to train our data, I used batch size of 20 epochs to create hyperparameters to harmonize during deep learning for quicker results when it comes to time management. Results show that the learning rate is determined to be 0.0001. To this reate, I applyed Dense layer with two neurons for two output classes (benign and malignant) with activation function as a softmax. Also used is Adam optimizer for optimization. The results are below:

![DLT](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/Deep%20Learning%20Results.png)

### Results 4: Deep Learning Outcome
After  hyper tuning our data with parameters and deep learning pushed to the AI, I have successfully training our AI to classify breast cancer to show the differences between Malignant or Benign with a 98.87% success rate:

![Results2](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/Model%20Accuracy.png) 

This is after 

### Results 5: After Training

The following results show our accuracy of 98.87% after applying the binary-cross-entropy for loss function and Adam optimizer for optimization. We can see that the orange bar that represents testing, starts at 88% and jagged up to approximately 92%. The blue line, representing training, goes from 87% to 98.87% accuracy.

## Results 6. Data Exploration Results

The following results showcases data exploration to get to know our dataset (even though training our AI (Artificial Intelligence) is the main focus of this project).  I enjoy learning what the data holds and what story it tells us so we can focus on future ideas to increase the accuracy of our AI.

### Results 7: Violin Plot

The first image is showcasing the median of texture_mean for Malignant and Benign.  The shape indicates the two are separated, and yet for fractal_dimension_mean, it's close together:

![ViolinPl1](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/violin_graph_median.JPG)

The second shape of the violin plot for area_se looks warped with the distribution points for benign and malignant being very different. This showcases that our AI is able to seperate the images according to thw two focuses (benign/malignat).  Variance looks highest for fractal_dimension_worst and concavity_worst with concave_points_worst seems to be similar data distribution.

![ViolinPl1](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/Violin_graph_median2.png)

### Results 8: Heatmap

The heatmap has a correlation of > 0.8, means, std errors and worst dimension lengths of compactness, concavity and points of the concave are high in correlation to one anter.  Our Mean, worst dimensions of radius, std errors, area and perimeter of tumors have a correlation of 1.  Results below:

![Heatmap](https://github.com/CrawleyM29/PractiumII-BCC/blob/data-engineering/Plots/Correlation%20heatmap_testing.JPG)

## 8. Summary

The deep learning AI training is a success and distinguishes which images are benign and malignant breast cancer from a combination of small imaging using Deep Learning Python with a 98.87% success rate and using exploratory data to understand our dataset for better results, and showcasing that our Malignant and Benign are great features after training.

## 9. References

Dataset: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
