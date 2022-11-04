import cv2
import numpy as np
import matplotlib.pyplot as plt

img256 = cv2.imread('E:/lena.jpg', 0)

def gray_transfer(gray_level,img):
    height, width = img.shape[:2]
    new_img = np.zeros((height, width))
    a = int(255 / gray_level)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = int(img[i][j] / a)
    
    return new_img

img2 = gray_transfer(2,img256)
img4 = gray_transfer(4,img256)
img8 = gray_transfer(8,img256)
img16 = gray_transfer(16,img256)
img32 = gray_transfer(32,img256)
img64 = gray_transfer(64,img256)
img128 = gray_transfer(128,img256)


plt.figure(figsize=(10, 18))

plt.subplot(241), plt.imshow(img2, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(242), plt.imshow(img4, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(243), plt.imshow(img8, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(244), plt.imshow(img16, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(245), plt.imshow(img32, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(246), plt.imshow(img64, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(247), plt.imshow(img128, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(248), plt.imshow(img256, 'gray'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()