#轮廓发现是基于图像边缘提取的基础寻找对象轮廓的方法，所以边缘提取的阈值选定会影响最终轮廓发现结果
#API介绍   findContours发现轮廓（原理：基于拓扑结构）   drawContours绘制轮廓
import cv2 as cv
import numpy as np


def contours_demo(image):
    #获取二值图像1
    # dst = cv.GaussianBlur(image,(3,3),0)
    # gray = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    # ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)#图像二值化
    # cv.namedWindow("binary image", cv.WINDOW_NORMAL)
    # cv.imshow("binary image",binary)


    #获取二值图像二
    dst = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray,cv.CV_16SC1,1,0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    binary = cv.Canny(xgrad,ygrad,50,150)
    cv.namedWindow("Canny Edge", cv.WINDOW_NORMAL)
    cv.imshow("Canny Edge", binary)

    countours,heriachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE,)
    #RETR_TREE返回轮廓的所有树形结构，RETR_EXTERNAL返回最外层的轮廓,CHAIN_APPROX_SIMPLE简单的提取
    #contours为轮廓存贮，heriachy为层次信息
    for i,contour in enumerate(countours):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        cv.drawContours(image,countours,i,(0,0,255),-1)#-1为填充轮廓  2为绘制轮廓宽度
        print(i)
    cv.namedWindow("detect_contours", cv.WINDOW_NORMAL)
    cv.imshow("detect_contours", image)






#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Aluminum_alloy\pic\pic/3-2.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_NORMAL)
#显示图片
cv.imshow("input image",src)
contours_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()