import math
import os.path
import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        continue
    hands, img = detector.findHands(img)
    if not hands:
        continue
    hand = hands[0]
    x, y, w, h = hand['bbox']
    x1, y1 = max(0, y - offset), y + h + offset
    x2, y2 = max(0, x - offset), x + w + offset

    imgData = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    imgCrop = img[x1:y1, x2:y2]

    if h > w:
        k = imgSize / h
        wCal = math.floor(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        gap = math.floor((300 - wCal) / 2)
        imgData[:, gap:wCal + gap] = imgResize
        prediction, index = classifier.getPrediction(img)
        print(prediction, index)
    else:
        k = imgSize / w
        hCal = math.floor(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        gap = math.floor((300 - hCal) / 2)
        imgData[gap:hCal + gap, :] = imgResize
        prediction, index = classifier.getPrediction(img)
        print(prediction,index)

    cv2.imshow('ImageData', imgData)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
