#图像特别大的情况下，一般进行分块进行二值化
import cv2 as cv
import numpy as np


def big_image_binary(image):
    print(image.shape)
    cw = 256
    ch = 256
    h,w = image.shape[:2]
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    for row in range(0,h,ch):
        for col in range(0,w,cw):
            roi = gray[row:row+ch,col:cw+col]
            #输出roi均值与标准方差，若均值与方差均接近于0，表明其为空白图像
            print(np.std(roi),np.mean(roi))#std 为方差   mean为均值
            dev = np.std(roi)
            if dev<15:
                gray[row:row + ch, col:cw + col] = 255
            else:
                #图像二值化   采用整体阈值化方法效果不佳
                ret,dst=cv.threshold(roi,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
                #采用局部阈值化方法
                dst=cv.adaptiveThreshold(roi,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blockSize=127,C=20)#blockSize  255*255的图像分为四块每块为127,消除20左右的噪声


                #当均值为255，方差为0的情况下，可以认定为空白图像，可以设计阈值，例如：当方差小于15的时候可以将dst中的值全部赋值为0，可以消除产生的噪声，可以消除阈值分割带来的失误


                gray[row:row+ch,col:cw+col]=dst

    cv.imwrite('F:\software\pic\Person/pangzi_binary.jpg',gray)#保存图片


#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Person/pangzi.jpg')
src1 = cv.imread('F:\software\pic\Aluminum_alloy/demo.png')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
big_image_binary(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()