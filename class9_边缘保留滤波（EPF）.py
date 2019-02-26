#高斯双边
#均值迁移

import cv2 as cv
import numpy as np


def bi_demo(image):
    dst = cv.bilateralFilter(image,0,60,10)#一般选择第二个参数为0，第三个参数sigmaColor大一点，第四个参数sigmaSpace小一点
    cv.imshow("bi_demo",dst)

def shift_demo(image):#均值迁移
    dst = cv.pyrMeanShiftFiltering(image,10,50)#一般选择第二个参数sp，第三个参数SR
    cv.imshow("shift_demo",dst)

#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow&opencv/opencv/pic/pangzi.jpg')
#opencv命名
# cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
# cv.imshow("input image",src)

bi_demo(src)
shift_demo(src)

#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()