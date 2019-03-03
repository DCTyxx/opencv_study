#霍夫直线变换介绍
#Hough Line Transform 用来做直线检测，前提条件-边缘检测已经完成，平面空间到极坐标空间转换



import cv2 as cv
import numpy as np

#直线检测
def line_dection(image):
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize=3) #canny 边缘检测窗口的大小
    lines = cv.HoughLines(edges,1,np.pi/180,150)#输入图片，角度变化为一度，threshold设定低值为200
                            #半径的步长为1
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)#将直线绘画到图片上
    cv.imshow("image_lines",image)


#直接显示出事直线的可能性
def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # canny 边缘检测窗口的大小
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 100,minLineLength=50,maxLineGap=10)  # 输入图片，角度变化为一度，threshold设定低值为200
    # 半径的步长为1                                   认为是是直线的最小取值，当线与线之间存在断点的个数
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 将直线绘画到图片上
    cv.imshow("line_detect_possible_demo", image)









#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Aluminum_alloy/demo.png')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
line_detect_possible_demo(src)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()