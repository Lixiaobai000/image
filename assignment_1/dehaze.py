from PIL import Image


import cv2
import numpy as np
import argparse
 
# 图片相关参数：
# 1.默认图片
defeat_photo = r'..\Input\haze\canon3.bmp'
# 2.待处理图片
photo_name = 'trees.png'
# 3.待处理图片所在目录地址
ImgInput = r'..\Input\haze\{}'.format(photo_name)
# 4.处理后图片保存地址
ImgFile = r'..\Output\HazeRemove'
 
 
# 计算雾化图像的暗通道
def DarkChannel(img, size=15):
    """
    暗通道的计算主要分成两个步骤:
    1.获取BGR三个通道的最小值
    2.以一个窗口做MinFilter
    ps.这里窗口大小一般为15（radius为7）
    """
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img
 
 
# 估算全局大气光值
def GetAtmo(img, percent=0.001):
    """
    1.计算有雾图像的暗通道
    2.用一个Node的结构记录暗通道图像每个像素的位置和大小，放入list中
    3.对list进行降序排序
    4.按暗通道亮度前0.1%(用percent参数指定百分比)的位置，在原始有雾图像中查找最大光强值
    """
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)
 
 
# 估算透射率图
def GetTrans(img, atom, w):
   
    x = img / atom
    t = 1 - w * DarkChannel(x, 15)
    return t
 
 
def GuidedFilter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    # 1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    # 2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # 3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    # 4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # 5
    q = mean_a * i + mean_b
    return q
 
 
# 主程序
def DeHaze():
    path, output, photo, t0, w = opt.input, opt.output, opt.photo, opt.threshold_value, opt.dehaze_degree
    # 读取待处理图像
    im = cv2.imread("E:/3.jpeg")
     #im=cv2.imread("E:/Wu.jpeg")
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255
 
    atom = GetAtmo(img)
    trans = GetTrans(img, atom, w)
    trans_guided = GuidedFilter(trans, img_gray, 20, 0.0001)

    trans_guided = cv2.max(trans_guided, t0)
 
    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
 
    # 显示&保存结果
    cv2.imshow("source", img)
    cv2.imshow("result", result)
    cv2.waitKey()
    if output is not None:
        cv2.imwrite("{}/{}".format(output, photo), result * 255)
 
 
# 可通过命令行传递参数
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default=ImgInput)
parser.add_argument('-o', '--output', default=ImgFile)
parser.add_argument('-p', '--photo', default=photo_name)

parser.add_argument('-t', '--threshold_value', default=0.25)
#w为去雾程度，一般取0.95。w的值越大，去雾效果越明显
parser.add_argument('-w', '--dehaze_degree', default=0.95)
opt = parser.parse_args()
print(f'parser.parse_args(解析器的参数):\n{opt}')
 
# 执行主程序
if __name__ == '__main__':
    if opt.input is None:
        DeHaze(defeat_photo)
    else:
        DeHaze()
