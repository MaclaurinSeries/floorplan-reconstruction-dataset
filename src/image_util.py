import numpy as np
from PIL import Image
import cv2

def get_bounding_box(img):
    if img.shape[2] > 3:
        img = Image.fromarray(img)
        imgwhite = Image.new("RGB", img.size, (255, 255, 255))
        imgwhite.paste(img, mask=img.split()[3])
        img = np.array(imgwhite)
    
    x, y = np.where(np.all(img != [255, 255, 255], axis=2))
    return x[np.argmin(x)], y[np.argmin(y)], x[np.argmax(x)], y[np.argmax(y)]

def resize_image(img, height=-1, width=-1):
    if width == -1 and height == -1:
        return img
    
    if height == -1:
        height = int(width * img.shape[0] / img.shape[1])
    if width == -1:
        width = int(height * img.shape[1] / img.shape[0])
    
    nimg = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    
    return nimg

def remake_image(img, bg, scale=(1,1), shift=(0,0), W=-1, H=-1):
    mask = img[:,:,3] > 0
    
    shp = img.shape
    nshp = (int(shp[1] * scale[1]),
            int(shp[0] * scale[0]))
    new_img = cv2.resize(bg, dsize=nshp, interpolation=cv2.INTER_CUBIC)
    
    loc = (int((nshp[1] // 2 - shp[0] // 2) * (shift[0] + 1)),
           int((nshp[0] // 2 - shp[1] // 2) * (shift[1] + 1)))
    
    new_img[loc[0] : loc[0] + shp[0],
            loc[1] : loc[1] + shp[1],
            : ][mask] = img[:,:,:3][mask]
    
    translation = np.array([
        [1, 0, loc[1]],
        [0, 1, loc[0]],
    ])
    
    if W == -1 and H == -1:
        return new_img, translation
    
    resized_img = resize_image(new_img, H, W)
    H = resized_img.shape[0]
    W = resized_img.shape[1]
    
    translation = np.dot(np.array([
        [H / nshp[0], 0],
        [0, W / nshp[1]],
    ]), translation)
    
    return resized_img, translation

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

def similarity_feature(img1, img2):
    MIN_MATCHES = 50
    
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

        return corrected_img
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50],None, flags=2)

    return img3