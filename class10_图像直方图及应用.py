import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#汇出直方图(只统计频次的直方图)
def plot_demo(image):
    plt.hist(image.ravel(),256,[0,256])
    #image.ravel() 统计图片中各像素的频次，256为bin个数，[0，256]为范围
    plt.show()


#图片直方图,获得图片的特征
def image_demo(image):
    color = ("blue","green","red")
    for i ,color in enumerate(color):
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        hist = cv.calcHist([image],[i],None,[256],[0,256])
        plt.plot(hist,color=color)
        plt.xlim([0,255])
    plt.show()

    plt.hist(image.ravel(),256,[0,256])
    #image.ravel() 统计图片中各像素的频次，256为bin个数，[0，256]为范围
    plt.show()


#直方图的均衡化(opencv当中都是基于灰度图)
def equalHist_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)#直方图均衡化
    cv.imshow("equal",dst)


#局部直方图均衡化
def clahe_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)) #局部直方图均衡化
    dst = clahe.apply(gray)
    cv.imshow("clahe_demo",dst)



#直方图比较
def create_rgb_hist(image):
    h,w,c = image.shape
    rgbHist = np.zeros([16*16*16,1],np.float32)
    #r,g,b的bin的取值为0~16三个相乘及取值空间范围，1为一列
    bsize = 256/16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int((b/bsize)*16*16)+np.int((g/bsize)*16)+np.int(r/bsize)
            rgbHist[np.int(index),0] = rgbHist[np.int(index),0]+1
    return rgbHist

def hist_compare(image1,image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv.compareHist(hist1,hist2,cv.HISTCMP_BHATTACHARYYA)#巴氏距离  巴氏距离越大越不相似
    match2 = cv.compareHist(hist1,hist2,cv.HISTCMP_CORREL)#相关性  相关性越小越不相似
    match3 = cv.compareHist(hist1,hist2,cv.HISTCMP_CHISQR)#卡方  卡方大说明图片相差大
    print("巴氏距离：%s,相关性：%s,卡方：%s"%(match1,match2,match3))

#读取图片,读取的数据为numpy的多元数组
src1 = cv.imread('F:\software/tensorflow&opencv\opencv\pic/windows.jpg')
src2 = cv.imread('F:\software/tensorflow&opencv\opencv\pic/pangzi.jpg')

hist_compare(src1,src2)

#opencv命名
# cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image1",src1)
cv.imshow("input image2",src2)
# plot_demo(src)
# image_demo(src)
# equalHist_demo(src1)
# clahe_demo(src1)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()