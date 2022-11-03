
import cv2
import math
import numpy as np
 
img = cv2.imread("E:/1.jpg", 1)
imginfo = img.shape
height = imginfo[0]
width = imginfo[1]
mode = imginfo[2]
 
dst = np.zeros(imginfo, np.uint8)
 
 #向右平移200个像素
for i in range( height ):
  for j in range( width - 200 ):
    dst[i, j + 200] = img[i, j]
 
cv2.imshow('translation_result', dst)
cv2.waitKey(0)
   