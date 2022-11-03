import cv2 
from matplotlib import pyplot as plt
import numpy as np 

def histogram(img):
    rows = img.shape[0]
    cols = img.shape[1]
    
    htg = [ 0 for i in range(256)]
    for r in range(rows):
        for c in range(cols):
            v = img[r][c]
            htg[v] += 1
    return htg

def calcCdf(histogram, pixelCount):
    cdf = [0 for i in range(256)]
    cdf[0] = histogram[0] / pixelCount
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + histogram[i] / pixelCount
    return cdf   

def otsu(image):
    htg = histogram(image)
    
    pixelCount = image.shape[0] * image.shape[1]
    cdf = calcCdf(htg,pixelCount)

    sigma = [0 for i in range(255)]
    for t in range(1, 255):
        w_b = cdf[t]
        w_f = 1 - w_b
         
        #背景像素灰度和
        sum_b = 0
        p1 = 0
        for i in range(t):
            sum_b += i * htg[i]
            p1 += htg[i]

        # 前景像素灰度和
        p2 = 0
        sum_f = 0
        for j in range(t, 256):
            sum_f += j * htg[j]     
            p2 += htg[j]
        
        if p1 == 0 or p2 == 0:
            continue
        u_b = sum_b / p1
        u_f = sum_f / p2

        sigma[t] = w_f * w_b * ((u_f - u_b)**2)
    return sigma

if __name__ == '__main__':
    image = cv2.imread("E:/1.jpg", 0)
    htg = histogram(image)
    htg2 = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    sigma = otsu(image)
    t = sigma.index(max(sigma))
     

    plt.plot(sigma.index(max(sigma)), max(sigma), 'ks')
    showMax = "(" + str(int(sigma.index(max(sigma)))) + "," +  str(int(max(sigma))) + ")"
    plt.annotate(showMax, xy=(sigma.index(max(sigma)), max(sigma)))
    plt.plot(sigma)
    plt.title('sigma^2(t)')

    (T, dst) = cv2.threshold(image, t, 255,cv2.THRESH_BINARY)
    (T2,dst2) = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    print(T2)
    cv2.imshow("otsu",dst)
    cv2.waitKey(0)
    plt.show()