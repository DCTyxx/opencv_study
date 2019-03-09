#高斯模糊去噪方面比均值模糊方面更好
import cv2 as cv
import numpy as np


#确保参数在0~255之间
def clamp(pv):
    if pv>255:
        return 255
    if pv<0:
        return 0
    else:
        return pv
#添加高斯噪声
def gaussian_noise(image):
    h,w,c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0,20,3)
            b = image[row, col, 0]#blue
            g = image[row, col, 1]#green
            r = image[row, col, 2]#red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[0])
            image[row, col, 2] = clamp(r + s[0])
    cv.imshow("noise image",image)

#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)

gaussian_noise(src)

#高斯模糊   保留图片的主要特征
#dst = cv.GaussianBlur(src,(0,0),10)
dst = cv.GaussianBlur(src,(5,5),0) #(5,5)为卷积核,0为高斯滤波的 σ
cv.imshow("GaussianBlur",dst)
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()