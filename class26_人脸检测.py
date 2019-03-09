import cv2 as cv
import numpy as np


def face_detect_demo():
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('F:\software/tensorflow&opencv\opencv-master\data\haarcascades_cuda\haarcascade_frontalface_default.xml')#直连检测器 传入haar的数据 或者lbp
    face = face_detector.detectMultiScale(gray,1.02,1)#1.02为以1.02挪动   2表示相邻有几个框能检测出人脸
    for x,y,w,h in face:
        cv.rectangle(src,(x,y),(x+w,y+h),(0,0,255),3)#在原图上绘制
    cv.imshow("face_detect",src)




#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\opencv_others/bobo.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_NORMAL)
cv.namedWindow("face_detect",cv.WINDOW_NORMAL)
#显示图片
cv.imshow("input image",src)
face_detect_demo()
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()