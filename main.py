import os
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier

videoCapture = cv2.VideoCapture(1)
classifier = Classifier('BiohazardDetectorApp/keras_model.h5', 'BiohazardDetectorApp/labels.txt')

imgWasteList = []
pathFolderWaste = 'BiohazardDetectorApp/Resources/ClassificationImages'
pathList = os.listdir(pathFolderWaste)

for pathway in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, pathway), cv2.IMREAD_UNCHANGED))
    
    
    


while True:
    _, img = videoCapture.read()
    imgResize = cv2.resize(img, (300, 300))
    
    imgBackground = cv2.imread('BiohazardDetectorApp/Resources/Live Feed.png')
    
    prediction = classifier.getPrediction(img)  
    print(prediction)
    
    classID = prediction[1] 
    
    if classID !=0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID -1], (800, 100))
        
    

    
    imgBackground[100:100+300, 100:100+300] = imgResize
    # Displays the images here:
    # cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)
    
    
        
        
print('we got it gang')
