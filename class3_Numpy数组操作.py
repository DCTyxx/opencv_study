import numpy as np
import cv2 as cv


def access_pixels(image):
    print(image.shape)#获取图片的宽高
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("height : %s, width : %s, channels : %s"%(height,width,channels))#答应图片的宽高
    #对图片做处理
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row,col,c]
                image[row, col, c] = 255-pv
    cv.imshow("pixels.show",image)

def inverse(image):
    #像素取反，等同于 image[row, col, c] = 255-pv
    dst = cv.bitwise_not(image)
    cv.imshow(dst)

#新建一张图片
def create_image():
    #新建一个三通道的八位图
    img = np.zeros([400,400,3],np.uint8)
    #修改图片  图片显示成蓝色 0表示，对第0个通道赋值
    img[ : , : , 0] = np.ones([400,400])*255
    cv.imshow("new image",img)

    #reahape的操作
    m1 = np.ones([3,3],np.float32)
    m1.fill(123333)#对值进行填充
    print(m1)

    m2 = m1.reshape([1,9])#renshape可以更改维度
    print(m2)


# 读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
# opencv命名
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# 显示图片
cv.imshow("input image", src)

create_image()

# #计算程序运行时长
# t1 = cv.getTickCount()
# access_pixels(src)
# t2 = cv.getTickCount()
# print("time : %s ms"%((t2-t1)/cv.getTickFrequency())*1000) #输出运行时长，单位为秒

cv.waitKey(0)

# 释放所有的类层
cv.destroyAllWindows()