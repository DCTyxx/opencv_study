#二值化图像即把图片变为0~1
#图像二值化方法：全局阈值，局部阈值
#opencv 中图像二值化的方法   OTSU  Triangle  自动与手动  自适应阈值
import cv2 as cv
import numpy as np

#全局阈值
def threshold_demo(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)#变为灰度图
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)#cv.THRESH_BINARY根据阈值把图像进行二值化，cv.THRESH_OTSU二值化的方法
    #ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_TOZERO | cv.THRESH_OTSU)#截断效果  THRESH_TRIANGLE对于直方图有单个波峰的时候效果非常好，一开始用于分割生物学细胞图像
    #ret, binary = cv.threshold(gray, 174, 255, cv.THRESH_TOZERO )#手动指定阈值   截断效果
    print("threshold value %s"%ret)
    cv.imshow("binary",binary)


#局部阈值
def local_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 变为灰度图
    #cv.imshow("dst", gray)
    dst = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blockSize=25,C=10)#255位最大值   ADAPTIVE_THRESH_GAUSSIAN_C局部二值化方法   blockSize必须为奇数  C为像素块的均值，若其他的值减去均值>10确定为白色或黑色，可以防止局部的噪声影响
    dst2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,blockSize=25,C=10)
    cv.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", dst)
    cv.imshow("ADAPTIVE_THRESH_MEAN_C", dst2)


def custom_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 变为灰度图
    h,w = gray.shape[:2]
    m = np.reshape(gray,[1,w*h])# 将gray转变为一维数组
    mean = m.sum()/(w*h) #计算平均值
    print("mean",mean)
    ret,binary = cv.threshold(gray,mean,255, cv.THRESH_BINARY)
    cv.imshow("custom_threshold",binary)


#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Aluminum_alloy/demo.png')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
#local_threshold(src)
custom_threshold(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()