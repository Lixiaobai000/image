import cv2
import numpy as np
import matplotlib.pyplot as plt

def down_sample(img):
    height, width = img.shape[:2]
    new_img = np.zeros((height//2, width//2))
    new_img = img[::2, ::2]
    return new_img

img = cv2.imread('E:/lena.jpg', 0)

img_2 = down_sample(img)
img_4 = down_sample(img_2)
img_8 = down_sample(img_4)

plt.figure(figsize=(10, 10))
plt.subplot(221), plt.imshow(img, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_2, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_4, 'gray'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_8, 'gray'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()