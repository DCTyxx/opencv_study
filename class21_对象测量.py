#对象测量  弧长与面积，多边形拟合，几何矩计算
#弧长与面积，轮廓发现，计算每个轮廓的弧长和面积，像素单位
#工厂精度加工检测

def measure_object(image):
    dst = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(dst,0,255,cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
    #打印阈值
    print("threshold value : %s"%ret)
    cv.imshow("binary image",binary)
    contours,hireachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        area = cv.contourArea(contour)#计算面积
        x,y,w,h = cv.boundingRect(contour)#轮廓的外接矩形
        mm = cv.moments(contour)#计算几何矩
        type(mm)
        #计算原点的中心
        if (mm["m00"] == 0):  # this is a line
            cx,cy=0,0
        else:
            cx = mm['m10']/mm['m00']
            cy = mm['m01']/mm['m00']
            cv.circle(image,(np.int(cx),np.int(cy)),2,(0,255,255),-1)
            #对轮廓绘制外接矩形
            cv.rectangle(image,(x,y),(x+w,y+w),(0,0,255),2)


    cv.namedWindow("measure_contours", cv.WINDOW_NORMAL)
    cv.imshow("measure_contours", image)







import cv2 as cv
import numpy as np
#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
measure_object(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()