#-*- coding:utf-8 -*-

#也可以 import cv2 as cv ,使用时用cv代替cv2
import cv2
import sys, os, re, argparse, logging
from scripts.utils.utils import run
from scripts.utils.image_utils import *
from multiprocessing.dummy import Pool as ThreadPool


def load_formulas(filename):
    formulas = dict()
    with open(filename) as f:
        for idx, line in enumerate(f):
            formulas[idx] = line.strip()
    return formulas


def change_image_channels(image, image_path):
    if image.mode != 'RGB':
        image = image.convert("RGB")
        os.remove(image_path)
        image.save(image_path)
    return image



def resize():
    img_dir = "error_lab/images/02.png"
    img = cv2.imread(img_dir)
    # cv2.imwrite("error_lab/images/", img)
    im = cv2.resize(img, None, None, fx=1/2, fy=1/2, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("error_lab/do.png", im)

#
# img_dir = "error_lab/images/2.png"
# out_dir = "error_lab/images/02.png"
# crop_image_for_trian(img_dir, out_dir)

resize()
#

# gold_dir = "error_lab/images/"
# for root, dirs, files in os.walk(gold_dir):
#     for fl in files:
#         image_dir = os.path.join(gold_dir, fl)
#         image = Image.open(image_dir)
#         change_image_channels(image, image_dir)


# filename = "error_lab/src-test.txt"
# dir = "data/images/"
# resize()

# { \frac { \partial } { \partial T } } S T { \frac { \partial } { \partial T } }
# { \frac { \partial } { \partial T } } S T { \frac { \partial } { \partial T } }






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