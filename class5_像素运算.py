#可以调整颜色的亮度，调整对比度
import cv2 as cv
import numpy as np
#常见图像混合

#像素加法
def add_demo(m1,m2):
    dst = cv.add(m1,m2)
    cv.imshow("add_demo",dst)

#像素减法
def subtract_demo(m1,m2):
    dst = cv.subtract(m1,m2)
    cv.imshow("subtract_demo",dst)

#像素除法
def divide_demo(m1,m2):
    dst = cv.divide(m1,m2)
    cv.imshow("divide_demo",dst)

#像素除法
def multipy_demo(m1,m2):
    dst = cv.multiply(m1,m2)
    cv.imshow("multipy_demo",dst)

#计算图像的均值,方差,方差越大，图片之间的差异性越大，方差越小图片中的差异性越小（对比度越小）可以判定 当方差阈值小于某一个数值时为无效信息
def other(m1,m2):
    m1,dev1 = cv.meanStdDev(m1)
    m2,dev2 = cv.meanStdDev(m2)
    h,w = m1.shape[:2]
    print(m1)
    print(m2)

    print(dev1)
    print(dev2)

    img = np.zeros([h,w],np.uint8)
    m,dev = cv.meanStdDev(img)
    print(m)
    print(dev)

#算法运算与几何运算

def logic_demo(m1,m2):
    # 与运算
    dst = cv.bitwise_and(m1,m2)
    cv.imshow("Logic_demo_and",dst)
    #或运算
    dst = cv.bitwise_or(m1,m2)
    cv.imshow("Logic_demo_or", dst)
    # 非运算(非运算只操作一张图)
    dst = cv.bitwise_not(m1)
    cv.imshow("Logic_demo_not", dst)

#明亮对比度 c为对比度，b为亮度(在原先每个通道的基础上加上亮度)
def contrast_brightness_demo(image,c,b):
    h,w,ch = image.shape
    #构建空白图片
    blank = np.zeros([h,w,ch],image.dtype)
    #调整亮度及对比度
    dst = cv.addWeighted(image,c,blank,1-c,b)
    cv.imshow("con-br-i-demo",dst)

# 读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software/tensorflow-inception\images/vgg_face_id\pic/flowers/image_0074.jpg')
src2 = cv.imread('F:\software/tensorflow-inception\images/vgg_face_id\pic/flowers/image_0080.jpg')
print(src.shape)
print(src2.shape)
# opencv命名
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 显示图片
cv.imshow("input image", src)
cv.imshow("input image2", src2)


contrast_brightness_demo(src2,1.5,10)

add_demo(src,src2)
subtract_demo(src,src2)
divide_demo(src,src2)
multipy_demo(src,src2)
other(src,src2)
logic_demo(src,src2)
cv.waitKey(0)

# 释放所有的类层
cv.destroyAllWindows()