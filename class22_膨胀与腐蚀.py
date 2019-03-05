#膨胀，腐蚀，图像形态学最基本的操作
#图像形态学是图像处理学科的一个单独分支学科，灰度与二值图像处理中重要手段，是由数学的集合论等相关理论发展起来的学科
#通过膨胀与腐蚀可以通过指定元素的保留与删除获得图像
#膨胀的作用   对象大小增加一个像素（3*3），平滑对象边缘，减少或者填充对象之间的距离
#腐蚀的作用   用最小值替换中心像素，对象大小建晒1个像素（3*3），平滑对象边缘，弱化或者分割图像之间的半岛型连接
import cv2 as cv
import numpy as np


def erode_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

    cv.imshow("binary",binary)

    kernel = cv.getStructuringElement(cv.MORPH_ERODE,(3,3))#结构元素的像素(3,3)
    dst = cv.erode(binary,kernel)#腐蚀操作

    cv.imshow("erode_demo",dst)


def dilate_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

    cv.imshow("binary",binary)

    kernel = cv.getStructuringElement(cv.MORPH_ERODE,(3,3))#结构元素的像素(3,3)
    dst = cv.dilate(binary,kernel)#腐蚀操作

    cv.imshow("erode_demo",dst)

#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
#erode_demo(src)
dilate_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()