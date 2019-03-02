import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def back_projection_demo():
    sample = cv.imread('F:/software/tensorflow-inception/pic_test/image_0068.jpg')
    target = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
    roi_hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target,cv.COLOR_BGR2HSV)
    #show image
    cv.imshow("sample",sample)
    cv.imshow("target",target)

    #通过调整bin的个数 及调整[180,256] 使得效果增强，bin越多对每个像素细分地越厉害，导致反向投影结果产生碎片化
    roihist = cv.calcHist([roi_hsv],[0,1],None,[180,256],[0,180,0,256])
    #roihist = cv.calcHist([roi_hsv],[0,1],None,[32,32],[0,180,0,256])
    #归一化到0-255之间
    cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)

    #生成反向投影                                     该取值范围不会改变
    dst=cv.calcBackProject([target_hsv],[0,1],roihist,[0,180,0,256],1)
    cv.imshow('backprojectionDemo',dst)



#绘制2D直方图
def hist2d_demo(image):
    hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    #计算俩个通道，通道为0/1,宽为255，高为180
    hist = cv.calcHist([image],[0,1],None,[180,256],[0,180,0,256])
    #显示直方图       插值方式
    plt.imshow(hist,interpolation='nearest')
    #加入标题
    plt.title("2D Histogram")
    plt.show()


#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
#opencv命名
#cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
#cv.imshow("input image",src)
#hist2d_demo(src)
back_projection_demo()
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()