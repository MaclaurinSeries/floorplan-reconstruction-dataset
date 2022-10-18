import numpy as np
from PIL import Image
import cv2

def get_bounding_box(img):
    x, y = np.where(np.all(img != [255, 255, 255], axis=2))
    return x[np.argmin(x)], y[np.argmin(y)], x[np.argmax(x)], y[np.argmax(y)]

def resize_image(img, width=-1, height=-1):
    if width == -1 and height == -1:
        return img
    
    if width == -1:
        width = int(height * img.shape[0] / img.shape[1])
    if height == -1:
        height = int(width * img.shape[1] / img.shape[0])
    
    nimg = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    
    return nimg

def remake_image(img, bg, mask, scale=(1,1), shift=(0,0), W=-1, H=-1):
    xl, yl, xr, yr = get_bounding_box(img)
    
    shp = img.shape
    nshp = (int(shp[1] * scale[1]),
            int(shp[0] * scale[0]))
    new_img = cv2.resize(bg, dsize=nshp, interpolation=cv2.INTER_CUBIC)
    
    loc = (int((nshp[1] // 2) * (shift[0] + 1) - shp[0] // 2),
           int((nshp[0] // 2) * (shift[1] + 1) - shp[1] // 2))
    
    idx = (mask[xl : xr, yl : yr] == 1)
    new_img[loc[0] : loc[0] + xr - xl,
            loc[1] : loc[1] + yr - yl,
            : ][idx] = img[xl : xr, yl : yr][idx]
    
    if W == -1 and H == -1:
        return (new_img,
                (loc[0],
                 loc[1],
                 loc[0] + xr - xl,
                 loc[1] + yr - yl
                ))
    
    resized_img = resize_image(new_img, W, H)
    H = resized_img.shape[0]
    W = resized_img.shape[1]
    
    return (resized_img,
            (int(loc[0] * W / nshp[0]),
             int(loc[1] * H / nshp[1]),
             int((loc[0] + xr - xl) * W / nshp[0]),
             int((loc[1] + yr - yl) * H / nshp[1])
            ))

def open_rgb(path):
    img = Image.open(path)
    
    if len(img.getbands()) <= 3:
        return np.asarray(img)

    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    
    return np.asarray(background)

def open_rgba(path):
    img = Image.open(path)
    
    return np.asarray(img)