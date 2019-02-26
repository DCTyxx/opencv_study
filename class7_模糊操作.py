#模糊是卷积的一种表现
#均值模糊
#中值模糊
#自定义模糊

import cv2 as cv
import numpy as np
#均值模糊
def blur_demo(image):
    dst = cv.blur(image,(1,3))#(1,3)为水平方向上与垂直方向上的模糊程度
    cv.imshow("blur_demo",dst)

#中值模糊 可以去除椒盐噪声
def median_blur_demo(image):
    dst = cv.medianBlur(image,5)
    cv.imshow("median_blur_demo",dst)

#自定义模糊
def custom_blur_demo(image):
    kernel = np.ones([5,5],np.float32)/25#(均值模糊的话/25 （25=5*5）)
    dst = cv.filter2D(image,-1,kernel=kernel)
    #filter2D参数 src为输入的图片，ddepth为默认参数,kernel为自定义的卷积核算子，dst为输出结果，anchor为锚点，delta，bordertype为边缘填充模式
    cv.imshow("custom_blur_demo",dst)

#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)

blur_demo(src)
median_blur_demo(src)
custom_blur_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()
