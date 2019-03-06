#开操作   图像形态学的重要操作之一，基于膨胀与腐蚀操作组合形成的，主要是应用在二值图像分析中，灰度图亦可，开操作=腐蚀+膨胀，输入图像+结构元素
#闭操作   图像形态学的重要操作之一，基于膨胀与腐蚀操作组合形成的，主要是应用在二值图像分析中，灰度图亦可，闭操作=膨胀+腐蚀，输入图像+结构元素
#开闭操作作用，去除小的干扰块-开操作，填充闭合区域-闭操作，水平或者垂直线提取
import cv2 as cv
import numpy as np


def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary",binary)
    kernel = cv.getStructuringElement(cv.MORPH_ERODE,(5,5))#矩形结构元素
    # kernel = cv.getStructuringElement(cv.MORPH_ERODE, (15, 1))  # (15, 1)矩形结构元素，去除垂直线，提取水平线，(1,15)去除水平线，提取垂直线
    binary = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)#开操作
    cv.imshow("open-result",binary)

def close_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary",binary)
    kernel = cv.getStructuringElement(cv.MORPH_ERODE,(5,5))#矩形结构元素
    binary = cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)#开操作
    cv.imshow("open-result",binary)



#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
open_demo(src)
close_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()