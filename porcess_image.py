#-*- coding:utf-8 -*-

#也可以 import cv2 as cv ,使用时用cv代替cv2
import cv2


def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()
    return formulas

filename = "error_lab/src-test.txt"
dir = "data/images/"
file = load_formulas(filename)[0]
img = cv2.imread(dir+file)
cv2.imwrite("error_lab/"+file, img)
im = cv2.resize(img, None, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imwrite("error_lab/do.png", im)



#
# #读入图片
# img=cv2.imread(filename)	#默认打开，彩色图像
# img_gray=cv2.imread(filename,0)#灰度图打开，黑白图像
#
# #显示图片
# cv2.imshow("Img",img)
# cv2.imshow("Img_gray",img_gray)
#
# #使图片长时间停留，不闪退
# cv2.waitKey(0)
#
# #保存图片
#
#
# #摧毁所有窗口
# cv2.destroyAllWindows()