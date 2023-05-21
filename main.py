import os
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier

liveFeed = cv2.VideoCapture(1)
classifier = Classifier('BiohazardDetectorApp/keras_model.h5', 'BiohazardDetectorApp/labels.txt')
#imports waste pictures
imgWasteList = []
pathFolderWaste = 'BiohazardDetectorApp/Resources/ClassificationImages'
pathList = os.listdir(pathFolderWaste)

for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))
#imports bins   
imgBinsList = []
pathFolderBins = 'BiohazardDetectorApp/Resources/Hazardous Check'
pathList = os.listdir(pathFolderBins)

for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins, path), cv2.IMREAD_UNCHANGED))   
    


classDictionary = {0:None,
            1:0,
            2:0,
            3:1,
            4:1}

while True:
    _, img = liveFeed.read()
    imgResize = cv2.resize(img, (300 , 300))
    
    backGround = cv2.imread('BiohazardDetectorApp/Resources/Live Feed.png')
    
    
    prediction = classifier.getPrediction(img)  
    print(prediction)
    
    classID = prediction[1] 
    
    if classID !=0:
        backGround = cvzone.overlayPNG(backGround, imgWasteList[classID-1], (800, 150))
        backGround = cvzone.overlayPNG(backGround, imgBinsList[classDictionary[classID]], (800, 500))
    

    
    backGround[100:100+300, 75:75+300] = imgResize
    
    cv2.imshow("Output", backGround)
    cv2.waitKey(1)
