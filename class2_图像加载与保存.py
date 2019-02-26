import cv2 as cv
import numpy as np


#读取视屏
def video_demo():
    capture = cv.VideoCapture(0)#打开电脑端的摄像头
    while(True):
        ret,frame=capture.read()#返回值ret,视屏的每一帧frame
        frame = cv.flip(frame,1)#镜像反转
        cv.imshow("video",frame)
        c = cv.waitKey(50)
        if c == 27:
            break


def get_image_info(image):
    print(type(image))#答应image的类别
    print(image.shape)#打印图片的长宽高,通道数目
    print(image.size)#打印图片的大小
    print(image.dtype)#打印位数
    plex_data = np.array(image)#打印像素
    print(plex_data)



# 读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
# opencv命名
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 显示图片
cv.imshow("input image", src)

get_image_info(src)
#video_demo()

gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)#获得灰度图片
#写入图片
cv.imwrite("F:\software/tensorflow&opencv\opencv\pic/demo.png",gray)
# 等待用户响应
cv.waitKey(0)

# 释放所有的类层
cv.destroyAllWindows()