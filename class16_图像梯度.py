#自定义卷积算子，实现图像的锐化以及模糊
#一阶导水与Soble算子       像素值的差异  差异值越大边缘越趋于明显
#二阶导数与拉普拉斯算子    在二阶导数的时候，最大变化处的值为零即边缘是零值，通过二阶导数计算，依据此理论可以计算图像二阶导数，提取边缘

import cv2 as cv
import numpy as np


def lapalian_demo(image):#拉普拉斯算子
    #dst=cv.Laplacian(image,cv.CV_32F)
    #lpls=cv.convertScaleAbs(dst)


    #自己定义算子
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    #增强型 拉普拉斯算子
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    dst = cv.filter2D(image,cv.CV_32F,kernel=kernel)#filter2D用内核卷积图像
    lpls = cv.convertScaleAbs(dst)

    cv.imshow("lapalian_demo",lpls)


def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)#求绝对值后转到8位的图像上去
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient-x",gradx)
    cv.imshow("gradient-y",grady)

    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow("gradient",gradxy)


def scharr_demo(image):  #Scharr算子为sobel算子的加强
    grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
    grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)#求绝对值后转到8位的图像上去
    grady = cv.convertScaleAbs(grad_y)
    cv.imshow("gradient-scharr-x",gradx)
    cv.imshow("gradient-scharr-y",grady)

    gradxy = cv.addWeighted(gradx,0.5,grady,0.5,0)
    cv.imshow("gradient-scharr",gradxy)



#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Aluminum_alloy/pic/demo.png')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
sobel_demo(src)
scharr_demo(src)
lapalian_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()