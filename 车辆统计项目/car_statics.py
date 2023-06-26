"""车辆统计项目"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
1. 加载视频
2. 背景减除（background subtract）预处理步骤，大多数计算机视觉项目都要用
3. 形态学识别车辆
4. 车辆统计
"""
cap = cv.VideoCapture('video.mp4')
# cap = cv.VideoCapture(0)
kernal = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

min_w = 100
min_h = 80

line_h = 550
offset = 5

num = 0

# 背景减除算法。创建mog对象。
mog = cv.createBackgroundSubtractorMOG2()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 把原始帧进行灰度化，然后高斯去噪
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 5)
    # 前景掩码，用在frame这一帧图片上。
    fgmask = mog.apply(blur)
    # 腐蚀
    erode = cv.erode(fgmask, kernal)
    # 膨胀
    dilate = cv.dilate(erode, kernal, iterations=2)
    # 闭运算，消除内部小方块。
    closed = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernal)

    # 画线
    cv.line(frame, (30, line_h), (1250, line_h), (255, 0, 0), 2)
    # 显示文字统计信息。
    cv.putText(frame, 'car num:', (500, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # 查找轮廓
    contours, hierarchy = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        if (w > min_w) and (h > min_h):
            cv.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            # 绘制轮廓中心点
            centerx = x + w / 2
            centery = y + h / 2
            # cv.circle(frame, (int(centerx), int(centery)), 5, (0, 255, 0), -1)
            # 判断中心点是否在线范围内
            if (centery < (line_h + offset)) and (centery > (line_h - offset)):
                num += 1
                # print(num)
                # cv.line
    cv.putText(frame, str(num), (800, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv.imshow('video', frame)
    key = cv.waitKey(1000 // 30)
    # esc的ASCLL码为27
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()

