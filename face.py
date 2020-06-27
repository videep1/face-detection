
import numpy as np
import cv2
from matplotlib import pyplot as plt
#create an object of cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img1 = cv2.imread('disha.jpg', 1)          
img2 = cv2.imread('sara.jpg',1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)

for (x,y,w,h) in faces1:
    cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray1[y:y+h, x:x+w]
    roi_color = img1[y:y+h, x:x+w]
cv2.imshow('img1',img1)

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)

for (x,y,w,h) in faces2:
    cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray1 = gray2[y:y+h, x:x+w]
    roi_color1 = img2[y:y+h, x:x+w]
cv2.imshow('img2',img2)
sift = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(roi_gray,None)
kp2, des2 = sift.detectAndCompute(roi_gray1,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.

k=len(good)
print(k)
img1=cv2.putText(img1,'2',(0,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow("score",img1)
cv2.waitKey(0)
cv2.destroyAllWindows()