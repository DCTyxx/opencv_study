import cv2 as cv
import numpy as np

#过滤某个颜色
def extrace_object():
    #读取视屏
    capture = cv.VideoCapture("F:\software/tensorflow&opencv\opencv/video/01.mp4")
    while (True):
        ret,frame = capture.read()
        if ret == False:
            break

        #过滤绿色   可以很好地追踪颜色对象
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

        lower_hsv = np.array([37,43,46])#HSV三通道的最小值
        upper_hsv = np.array([77, 255, 255])#HSV三通道的最大值

        mask = cv.inRange(hsv,lowerb=lower_hsv,upperb=upper_hsv)

        dst = cv.bitwise_and(frame,frame,mask=mask)
        cv.imshow("mask",mask)
        cv.imshow("dst", dst)

        cv.imshow("video",frame)
        c = cv.waitKey(40)
        if c==27:
            break




#色彩空间相互转换
def color_space_demo(image):
    #转换成GRAY的色彩空间
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    cv.imshow("gray",gray)
    #转换成HSV的色彩空间
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    cv.imshow("hsv",hsv)
    #转换成yuv色彩空间
    yuv = cv.cvtColor(image,cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)


# 读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
# opencv命名
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 显示图片
cv.imshow("input image", src)

#将RGB三通道分离
b,g,r = cv.split(src)
cv.imshow("Blue",b)
cv.imshow("Green",g)
cv.imshow("Red",r)

#更改通道，将第三通道更改为0
src[:,:,2]=0
cv.imshow("changed1 image",src)
#合并通道
src = cv.merge([b,g,r])
cv.imshow("changed image",src)

extrace_object()

cv.waitKey(0)

# 释放所有的类层
cv.destroyAllWindows()