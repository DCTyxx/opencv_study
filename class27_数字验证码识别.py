import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract as tess



def recognize_text():
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)#二值化  灰度二值化之后效果很好的不需要继续处理
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,2))#开操作
    bin1 = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,1))#开操作
    bin1 = cv.morphologyEx(bin1,cv.MORPH_OPEN,kernel)
    cv.imshow("binary-image",bin1)

    cv.bitwise_not(bin1,bin1)#底色转换  黑->白
    textImage = Image.fromarray(bin1)

    text = tess.image_to_string(textImage)
    print("识别结果：%s"%text)






#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\opencv_others\yanzheng.png')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
cv.imshow("input image",src)
recognize_text()
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()