
import cv2 as cv
import matplotlib.pyplot as plt
 
 
# 封装图片显示函数
def image_show(image):
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(image)
    plt.show()
 
 
# 迭代法阈值分割算法步骤：
# （1）选取初始分割阈值，通常可选图像灰度的平均值T。
# （2）根据阈值T将图像像素分割为背景和前景，分别求出两者的平均灰度T0和T1。
# （3）计算新的阈值T'=(T0 + T1) / 2。
# （4）若T == T‘，则迭代结束，T即为最终阈值。
# （5）否则令T = T’，转入第（2）步。
 
 
if __name__ == '__main__':
 
    # 读取灰度图像
    img_desk = cv.imread('E:/1.jpg', 0)
 
    # 求取平均灰度值
    T = img_desk.mean()
 
    # 迭代运算求取阈值
    while True:
 
        # 获取灰度图背景平均值
        T0 = img_desk[img_desk < T].mean()
 
        # 获取灰度图前景平均值
        T1 = img_desk[img_desk >= T].mean()
 
        # 计算新的阈值
        T_new = (T0 + T1) / 2
 
        # 判断是否终止
        if abs(T_new - T) < 0.1:
            break
        else:
            T = T_new
 
    # 阈值取整
    T = int(T)
 
    # 输出迭代的阈值
    print("最佳阈值 = ", T)
 
    # 根据最佳阈值进行图像分割
    _, img_bin = cv.threshold(img_desk, T, 255, cv.THRESH_BINARY)
 
    # 显示图像
image_show(img_bin)

