#canny算法介绍
#canny边缘检测分为五步
#通过高斯模糊去掉噪声，通过对图像求取梯度，根据图像的角度实现非最大信号实现压制，进行高低阈值的过滤
#高斯模糊-GaussianBlur  灰度转换-cvtColor   计算题都-Sobel/Scharr   非最大信号抑制     高低阈值输出二值图像
#T1,T2为阈值，凡是高于T2的都保留，凡是小于T1都丢弃，从高于T2的像素出发，凡是大于T1而且互相连接的，都保留，最终得到一个输出二值图像
#推荐的高低阈值比值为T2:T1 = 3:1/2:1其中T2为高阈值，T1为低阈值

import cv2 as cv
import numpy as np



def edge_demo(image):
    blurred = cv.GaussianBlur(image,(3,3),0)#高斯模糊 size如果有值，sigma不用设置 canny对噪声敏感
    gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    #x 方向的梯度
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)#canny不接收浮点数，所以推荐为cv.CV_16SC1
    #y 方向的梯度
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

    edge_output = cv.Canny(xgrad,ygrad,50,150)#50为低阈值，150为高阈值
    edge_output = cv.Canny(gray, 50, 150)

    cv.imshow("Canny Edge",edge_output)

    #输出为彩色
    dst = cv.bitwise_and(image,image,mask=edge_output)
    cv.imshow("Color Edge",dst)


#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Aluminum_alloy/demo.png')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
edge_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()