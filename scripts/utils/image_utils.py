import PIL
from PIL import Image
import numpy as np
import os
import cv2

def crop_image(img, output_path, default_size=None):
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8) # height, width
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        if not default_size:
            old_im.save(output_path)
            return False
        else:
            assert len(default_size) == 2, default_size
            x_min,y_min,x_max,y_max = 0,0,default_size[0],default_size[1]
            old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
            old_im.save(output_path)
            return False
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))

    old_im.save(output_path)
    return True

def change_image_channels(image, image_path):
    if image.mode != 'RGB':
        image = image.convert("RGB")
        # os.remove(image_path)
        image.save(image_path)
    return image



def crop_image_for_trian(img, output_path, default_size=None):
    old_im = Image.open(img).convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8) # height, width
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        if not default_size:
            old_im.save(output_path)
            return False
        else:
            assert len(default_size) == 2, default_size
            x_min,y_min,x_max,y_max = 0,0,default_size[0],default_size[1]
            old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
            old_im.save(output_path)
            return False
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])

    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))

    lap_5 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # 拉普拉斯5的锐化
    lap_9 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # 拉普拉斯9的锐化

    change_image_channels(old_im, output_path)

    src = cv2.imread(output_path)
    dst = cv2.filter2D(src, cv2.CV_8U, lap_5)

    im = Image.fromarray(dst)

    im.save("error_lab/images/oo.png")
    return True


def pad_group_image(img, output_path, pad_size, buckets):
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = pad_size
    old_im = Image.open(img)
    old_size = (old_im.size[0]+PAD_LEFT+PAD_RIGHT, old_im.size[1]+PAD_TOP+PAD_BOTTOM)
    j = -1
    for i in range(len(buckets)):
        if old_size[0]<=buckets[i][0] and old_size[1]<=buckets[i][1]:
            j = i
            break
    if j < 0:
        new_size = old_size
        new_im = Image.new("RGB", new_size, (255,255,255))
        new_im.paste(old_im, (PAD_LEFT,PAD_TOP))
        new_im.save(output_path)
        return False
    new_size = buckets[j]
    new_im = Image.new("RGB", new_size, (255,255,255))
    new_im.paste(old_im, (PAD_LEFT,PAD_TOP))

    new_im.save(output_path)
    return True

def downsample_image(img, output_path, ratio):
    assert ratio>=1, ratio
    if ratio == 1:
        return True
    old_im = Image.open(img)
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))

    new_im = old_im.resize(new_size, PIL.Image.LANCZOS)
    new_im.save(output_path)
    return True
