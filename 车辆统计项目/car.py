import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


cap = cv.VideoCapture('video.mp4')
mop = cv.createBackgroundSubtractorMOG2()
kernal = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 5)
    fg = mop.apply(blur)
    erode = cv.erode(fg, kernal)
    dilate = cv.dilate(erode, kernal, iterations=2)
    close = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernal)

    cv.line(frame, (30, 550), (1250, 550), (255, 0, 0), 2)

    contours, h = cv.findContours(close, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if w > 100 and h > 80:
            cv.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            centerx = int(x + w / 2)
            centery = int(y + h / 2)
            cv.circle(frame, (centerx, centery), 5, (255, 0, 0), -1)
            if (centery < 555) and (centery > 545):
                num += 1

    cv.putText(frame, 'car num:' + str(num), (800, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv.imshow('video', frame)

    key = cv.waitKey(int(1000 // 30))
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()


