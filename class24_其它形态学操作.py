#顶帽 计算原图与开操作之间的差值图像
#黑帽 计算闭操作与原图之间的差值图像
#形态学梯度
#基本梯度   基本梯度是用膨胀后的图像减去腐蚀后的图像得到差值图像，成为梯度图像也是opeencv中支持的计算形态学梯度的方法，而此方法得到梯度有被称为基本梯度
#内部梯度   使用原图像减去腐蚀之后的图像得到的差值图像，称为图像的内部梯度
#外部梯度   图像膨胀之后再减去原来的图像得到的差值图像，称为图像的外部梯度


import cv2 as cv
import numpy as np


#  普通图像的，黑帽   顶帽
def top_gray_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_ERODE,(5,5))#结构元素
    dst = cv.morphologyEx(gray,cv.MORPH_TOPHAT,kernel)#图像，选择的形态学操作，结构元素 顶帽操作
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)#图像，选择的形态学操作，结构元素 黑帽操作
    cimage = np.array(dst.shape,np.uint8)
    cimage = 25
    dst = cv.add(dst,dst)#由于图像不清晰    需要增加亮度
    cv.imshow("tophat",dst)


#  二值图像的，黑帽   顶帽
def hat_gray_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ERODE,(5,5))#结构元素
    #dst = cv.morphologyEx(binary,cv.MORPH_TOPHAT,kernel)#图像，选择的形态学操作，结构元素 顶帽操作
    dst = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)#图像，选择的形态学操作，结构元素 黑帽操作
    dst = cv.add(dst,binary)#由于图像不清晰    需要增加亮度
    cv.imshow("top",dst)

#图像的梯度 梯度
def hat_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ERODE,(3,3))#结构元素
    #dst = cv.morphologyEx(binary,cv.MORPH_TOPHAT,kernel)#图像，选择的形态学操作，结构元素 顶帽操作
    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)#图像，选择的形态学操作，结构元素 黑帽操作  MORPH_GRADIENT 为 梯度
    dst = cv.add(dst,binary)#由于图像不清晰    需要增加亮度
    cv.imshow("top",dst)

def hat_2_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_ERODE, (3, 3))  # 结构元素
    dm = cv.dilate(image,kernel)#使用特定的结构元素来扩展图像
    em = cv.erode(image,kernel)
    dst1 = cv.subtract(image,em) #内部梯度
    dst2 = cv.subtract(dm,image) #外部梯度
    cv.imshow("internal",dst1)
    cv.imshow("exterhate",dst2)


#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Aluminum_alloy\pic\pic/3-2.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
hat_2_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()