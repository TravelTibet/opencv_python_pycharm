# _*_ coding: utf-8 _*_
#
# Copyright (C) 2024 - 2024 xiname All Rights Reserved 
#
# @Time    : 2024/12/7 15:33
# @Author  : xiname
# @File    : chapter15.py
# @IDE     : PyCharm

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread("./pic/lena512g.bmp", 0)
# template = cv2.imread("./pic/temp.bmp", 0)
# th, tw = template.shape
# rv = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
# minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(rv)
# topLeft = minLoc
# bottomRight = (topLeft[0] + tw, topLeft[1] + th)
# cv2.rectangle(img, topLeft, bottomRight, 255, 2)
# plt.subplot(121), plt.imshow(rv, cmap='gray')
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(img, cmap='gray')
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# plt.show()


# import numpy as np
#
# # a = np.array([3,6,8,12,88])
# a = np.array([[3, 6, 8, 77, 66], [1, 2, 88, 3, 98], [11, 2, 67, 5, 2]])
# b = np.where(a > 5)
# print(b)


# x = [1, 2, 3]
# y = [4, 5, 6]
# z = [7, 8, 9]
# t = [x, y, z]
# print(t)
# # for i in zip(t):
# for i in zip(*t):
#     print(i)


# 交换位置

# loc = ([1, 2, 3, 4], [11, 12, 13, 14])
# print(loc, loc[::-1], sep="\n")


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("./pic/lena4.bmp", 0)
template = cv2.imread("./pic/lena4Temp.bmp", 0)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 1)
plt.imshow(img, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()
