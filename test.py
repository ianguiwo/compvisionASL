import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import keyboard
import time
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 125)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# will be above  #camera
global flag
flag = False

def speak_to_me(words):
    engine.say(' '.join(words))
    engine.runAndWait()
def display_text(words, flag):
    nw = labels[index]

    # if a word is detected not splicing is required
    if len(nw) > 1:
        if keyboard.is_pressed('a'):
            words.append(nw)

    # flag is default false, when flag is true letters will build a word a ---> an in the list
    # when flag is false a new word will begin a ----> a n
    elif flag:
        if keyboard.is_pressed('a'):
            words[-1] += nw
    else:
        if keyboard.is_pressed('a'):
            flag = True
            words.append(nw)
    return words, flag

#camera.. use videocapture(0) for default cam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/Goodbye"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "I Love You", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
          "Sorry", "T", "Thank You", "U", "V", "W", "X", "Y", "Z"]

frames = 0
global words
words = [""]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if frames<= 25:
        frames+=1

    else:
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

            aspectRatio = h/w
            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal)/2)
                imgWhite[:, wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw = False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw = False)

            cv2.putText(imgOutput, labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
            cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (255,0,255), 4)

            words, flag= display_text(words,flag)
            cv2.putText(imgOutput, ' '.join(words), (100, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            if (len(words) > 3 and not flag):
                speak_to_me(words)
                words = []
            elif keyboard.is_pressed('z'):
                words[-1] = ''
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        # else statement for if hands:
        else:
            if keyboard.is_pressed(' '):
                flag = False
            cv2.putText(imgOutput, ' '.join(words), (100, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)

