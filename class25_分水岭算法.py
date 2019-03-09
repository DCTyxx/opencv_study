#分水岭变换
#基于距离变换
#分水岭的变换     输入图像->灰度化（消除噪声）->二值化->距离变换->寻找种子->生成Marker->分水岭变换->输出图像->end
import cv2 as cv
import numpy as np

def watershed_demo():
    print(src.shape)
    blurred = cv.pyrMeanShiftFiltering(src,10,100)#边缘保留滤波
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    ret,binary = cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)#二值图像
    cv.namedWindow("binary-image", cv.WINDOW_NORMAL)
    cv.imshow("binary-image",binary)

    #形态学操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))#结构元素
    mb = cv.morphologyEx(binary,cv.MORPH_OPEN,kernel,iterations=2)#连续俩次进行开操作
    sure_bg = cv.dilate(mb,kernel,iterations=3)#使用特定的结构元素来扩展图像 拓展三次
    cv.namedWindow("mor-opt", cv.WINDOW_NORMAL)
    cv.imshow('mor-opt',sure_bg)

    #距离变化
    dist = cv.distanceTransform(mb,cv.DIST_L2,3)#L2 为欧几里得距离，为此移动大小为3
    dist_output = cv.normalize(dist,0,1.0,cv.NORM_MINMAX)#规范化数组的范数或值范围 当normType = NORM_MINMAX时（仅适用于密集数组）
    #cv.imshow("distance-t",dist_output*50)#显示距离变化的结果

    ret,surface = cv.threshold(dist,dist.max()*0.6,255,cv.THRESH_BINARY)#该函数通常用于从灰度图像中获取双级（二值）图像
    #获得种子
    #cv.imshow("surface-bin",surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg,surface_fg)#除了种子以外的区域
    ret,markers = cv.connectedComponents(surface_fg)#连通区域
    print(ret)

    #分水岭变化
    markers = markers+1
    markers[unknown==255] = 0
    markers = cv.watershed(src,markers=markers)#分水岭
    src[markers==-1]=[0,0,255] #将值为0的区域  赋值为255
    cv.namedWindow("result", cv.WINDOW_NORMAL)
    cv.imshow("result",src)




#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:\software\pic\Aluminum_alloy\pic\pic/3-2.jpg')
#opencv命名
cv.namedWindow("input image",cv.WINDOW_NORMAL)
#显示图片
cv.imshow("input image",src)
watershed_demo()
#等待用户响应
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()