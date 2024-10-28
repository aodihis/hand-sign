import math
import os.path
import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
imgFolder = 'data'
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
    else:
        k = imgSize / w
        hCal = math.floor(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        gap = math.floor((300 - hCal) / 2)
        imgData[gap:hCal + gap, :] = imgResize

    cv2.imshow('ImageData', imgData)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if 97 <= key <= 122:
        folder = f"{imgFolder}/{chr(key)}"
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for _ in range(100):
            cv2.imwrite(f"{folder}/img_{time.time()}.jpg", imgData)
            print(_)
