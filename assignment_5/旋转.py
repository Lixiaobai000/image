import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img = cv.imread("E:/1.jpg")

# 2 图像旋转
rows, cols = img.shape[:2]

# 2.1 生成旋转矩阵
M = cv.getRotationMatrix2D((cols/2, rows/2), 90, 1)  # 以图形的正中心作为旋转中心，旋转90°

# 2.2 进行旋转变换
dst1 = cv.warpAffine(img, M, (cols, rows))

M = cv.getRotationMatrix2D((cols/2, rows/2), 45, 0.5) # 旋转45°，缩小为原图的二分之一
dst2 = cv.warpAffine(img, M, (cols, rows))  # 调用warpAffine()函数完成图片旋转

# 3 图像展示
fig,axes=plt.subplots(nrows=1, ncols=3, figsize=(10, 8), dpi=100)
axes[0].imshow(img[:, :, ::-1])
axes[0].set_title("原图")
axes[1].imshow(dst1[:, :, ::-1])
axes[1].set_title("旋转后90°结果")
axes[2].imshow(dst2[:, :, ::-1])
axes[2].set_title("旋转后45°结果")
plt.show()
