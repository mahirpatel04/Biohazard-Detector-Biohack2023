import cv2
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(1)
classifier = Classifier('BiohazardDetectorApp/keras_model.h5', 'BiohazardDetectorApp/labels.txt')

imgBackground = cv2.imread('BiohazardDetectorApp/Resources/Live Feed.png')
while True:
    _, img = cap.read()
    prediction = classifier.getPrediction(img)  
    print(prediction)
    cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)
    
    
        
        
print('we got it gang')