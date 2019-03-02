#对特定场合有用（工业检测），对随机的场景作用比较小，模板匹配有一定的适应性
#模板匹配就是在整个图像区域发现与给定子图像匹配的小块区域
#所以末班匹配需要一个模板图像
#另外需要一个待检测的图像-原图像S，工作方法，在带检测图像上，从左往右，从上往下，计算模板图像与重叠子图像的匹配度，匹配程度越大，俩者相同的可能性越大
#匹配模板：TM_SQDIFF（平方不同，越小越相似）、TM_SQDIFF_NORMED（归一化平方不同）、TM_CCORR（相关性）、TM_CCORR_NORMED、TM_CCOEFF、TM_CCOEFF_NORMED

import cv2 as cv
import numpy as np


def template_demo():
    #目标图像与检测图像
    target = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
    tpl = cv.imread('F:/software/tensorflow-inception/pic_test/image_0066.jpg')
    cv.imshow("template image",tpl)
    cv.imshow('target',target)
    methods=[cv.TM_SQDIFF_NORMED,cv.TM_CCORR_NORMED,cv.TM_CCOEFF_NORMED]
    #获取模板的宽高
    th,tw = tpl.shape[:2]
    for md in methods:
        print(md)#打印出来看看
        result = cv.matchTemplate(target,tpl,md)#开始匹配   opencv将计算出来的值放在result当中
        min_val,max_val,min_loc,max_loc = cv.minMaxLoc(result)
        if (md == cv.TM_SQDIFF_NORMED):#如果是平方不同的话  取最小值
            tl = min_loc
        else:
            tl = max_loc

        #选择点位后，计算
        br = (tl[0]+tw,tl[1]+th)
        #将矩形绘制到原图上去       线的颜色 线宽
        cv.rectangle(target,tl,br,(0,0,255),2)
        cv.imshow("match-"+np.str(md),target)
        #cv.imshow("match-" + np.str(md), result)





#读取图片,读取的数据为numpy的多元数组
src = cv.imread('F:/software/tensorflow-inception/pic_test/image_0067.jpg')
#opencv命名
#cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
#显示图片
#cv.imshow("input image",src)
#等待用户响应
template_demo()
cv.waitKey(0)
#释放所有的类层
cv.destroyAllWindows()