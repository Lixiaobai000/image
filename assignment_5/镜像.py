import cv2
import numpy as np
 
 
img = cv2.imread('E:/1.jpg', 1)
cv2.imshow('src', img)
imginfo = img.shape
height= imginfo[0]
width = imginfo[1]
deep = imginfo[2]
 
dst = np.zeros([height*2, width, deep], np.uint8)
 
for i in range( height ):
  for j in range( width ):
    dst[i,j] = img[i,j]
    dst[height*2-i-1,j] = img[i,j]
 
for i in range(width):
  dst[height, i] = (0, 0, 255)
cv2.imshow('mirror_result', dst)
cv2.waitKey(0)