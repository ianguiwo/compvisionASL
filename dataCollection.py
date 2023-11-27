import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/A"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        #creates array of ones; copy of image 300x300, times 255 for white color
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        #crops image to fit hand, offset by 20 px
        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        #if value is above 1, height > width
        aspectRatio = h/w
        #if statement to make image fit white box; stretch height or width according to sign
        if aspectRatio > 1:
            k = imgSize/h
            #wCal = width calculated (w/ original height)...rounds value up
            wCal = math.ceil(k*w)
            #resize image
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            #calculates gap needed to center image
            wGap = math.ceil((imgSize - wCal)/2)
            # overlays camera feed on top of white image
            imgWhite[:, wGap:wCal+wGap] = imgResize

        #same thing but for width
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        #show image
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    #if "s" is pressed, image is saved
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)