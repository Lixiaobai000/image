import numpy as np
import cv2
def linear_transform(img):
    height,width = img.shape[:2]
    r1,s1 = 80,10
    r2,s2 = 140,200
    k1 = s1 / r1   # 第一段斜率
    k2 = (s2 - s1) / (r2 - r1) # 第二段斜率
    k3 = (255 - s2) / (255 - r2)  # 第三段斜率
    img_copy = np.zeros_like(img)
    for i in range(height):
        for j in range(width):
            if img[i,j] < r1 :
                img_copy[i,j] = k1 * img[i,j]
            elif r1 <= img[i,j] <= r2:
                img_copy[i,j] = k2 * (img[i,j] - r1) + s1
            else:
                img_copy[i,j] = k3 * (img[i,j] - r2) + s2
    return img_copy
img = cv2.imread('E:/1.jpg',0)
ret = linear_transform(img)
cv2.imshow('Piecewise_linear_result',ret)
cv2.waitKey()
cv2.destroyAllWindows()