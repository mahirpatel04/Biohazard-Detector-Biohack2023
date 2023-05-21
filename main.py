import os
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(1)
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
    
print(pathList)

classDictionary = {0:None,
             1:0,
             2:0,
             3:1,
             4:1}

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (300 , 300))
    
    imgBackground = cv2.imread('BiohazardDetectorApp/Resources/Live Feed.png')
    
    
    prediction = classifier.getPrediction(img)  
    print(prediction)
    
    classID = prediction[1] 
    
    if classID !=0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID-1], (800, 150))
        imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classDictionary[classID]], (800, 500))
    

    
    imgBackground[100:100+300, 75:75+300] = imgResize
    
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)
