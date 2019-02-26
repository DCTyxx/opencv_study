#泛洪填充，可以填充该区域内所有的点，或者是该区域内

#ROI区域即对某区域感兴趣的区域

import cv2 as cv
import numpy as np
#做色彩填充
def fill_color_demo(image):
    copyImg = image.copy()
    h,w = image.shape[:2]#获取图片的宽高
    mask = np.zeros([h+2,w+2],np.uint8)#mask大小固定
    cv.floodFill(copyImg,mask,(30,30),(0,255,255),(100,100,100),(50,50,50),cv.FLOODFILL_FIXED_RANGE)#FLOODFILL_FIXED_RANGE为色彩填充方式，当前像素全部填充
    #在(30,30的位置取像素，减去（100,100,100）获得他要填充的像素最低范围，加上（50,50,50）获得他要填充的像素的最大值范围)
    cv.imshow("Fill_color_demo",copyImg)

#二值填充
def fill_color_demo1():
    image = np.zeros([400,400,3],np.uint8)#新建一张图
    image[100:300,100:300,:]=255#图片上色
    cv.imshow("Fill_color_demo1",image)

    mask = np.ones([402,402,1],np.uint8)
    mask[101:301,101:301] = 0
    cv.floodFill(image,mask,(200,200),(0,0,255),cv.FLOODFILL_MASK_ONLY)
    cv.imshow("filled binary",image)

#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software/tensorflow-inception\images/vgg_face_id\pic/flowers/image_0098.jpg')
# src = cv.imread('F:\software/tensorflow-inception\images/vgg_face_id\pic/flowers/image_0074.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)

fill_color_demo1()
# fill_color_demo(src)
# #圈划ROI区域
# flower = src[50:250,100:300]
# gray = cv.cvtColor(flower,cv.COLOR_BGR2GRAY)#转为灰度图
# backflower = cv.cvtColor(gray,cv.COLOR_GRAY2BGR)#转为三色位图
# src[50:250,100:300] = backflower
# cv.imshow("flower",src)

#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()