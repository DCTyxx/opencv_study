#霍夫圆检测对噪声比较敏感，所以首先要对图像做中值滤波。
#基于效率考虑，Opencv中实现的霍夫变换圆检测是基于图像梯度的实现，分为两步：
#   检测边缘，发现可能的圆心
#   基于第一步的基础上从候选圆心开始计算最佳半径大小

import cv2 as cv
import numpy as np

def detect_hough_cricle_demo(image):
    dst = cv.pyrMeanShiftFiltering(image,10,100)# 图片  空间距离  颜色距离
    cimage = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage,cv.HOUGH_GRADIENT,1,20,param1=60,param2=50)#调节param1=60,param2=50的大小
    #第一个为图形片 第二个为方法 第四个为步长 第五个为 圆心小于20的为同一个圆，param1边缘提取的高值，param2边缘提取的低值
    circles = np.uint16(np.around(circles))#将数值变为int类型
    for i in circles[0,:]:
        cv.circle(image,(i[0],i[1]),i[2],(0,0,255),2)#绘画圆
        cv.circle(image,(i[0],i[1]),2,(255,0,0),2)#绘画圆心
        cv.imshow("circles",image)


#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\opencv_others\coin.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)

detect_hough_cricle_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()