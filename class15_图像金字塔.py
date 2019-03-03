#图像金字塔：reduce = 高斯模糊+降采样
#PyrDown 降采样
#PyrUp 还原   高斯金字塔与拉普拉斯金字塔


import cv2 as cv
import numpy as np


def pyramid_demo(image):#高斯金字塔
    level = 5
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        #降采样
        dst=cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramis_dowm_"+str(i),dst)
        temp = dst.copy()
    return pyramid_images


def lapalian_demo(image):#拉普拉斯金字塔
    pyramid_image=pyramid_demo(image)
    level = len(pyramid_image)
    for i in range(level-1,-1,-1):#从高到低，从level-1到-1每次step -1
        #原图并没有放到数组里，原图需要特殊处理
        if (i-1)<0:
            expand = cv.pyrUp(pyramid_image[i], dstsize=(image.shape[1],image.shape[0]))  # 还原，还原后，大小跟下一层大小相同
            lpls = cv.subtract(image, expand)  # l1=g1-expand(g2)  substarct为减法
            cv.imshow("lapalian_down_" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_image[i],dstsize=(pyramid_image[i-1].shape[1],pyramid_image[i-1].shape[0])) #还原，还原后，大小跟下一层大小相同,pyrup函数需要这些尺寸作为宽度x高度
            lpls = cv.subtract(pyramid_image[i-1],expand)#l1=g1-expand(g2)  substarct为减法
            cv.imshow("lapalian_down_"+str(i),lpls)


#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Aluminum_alloy/demo.png') #图片像素推荐为2的倍数
#src = cv.resize(src,(512,512))#图像变换
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
lapalian_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()