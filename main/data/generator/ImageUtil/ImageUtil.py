import numpy as np
from PIL import Image
from rasterio.features import shapes
from scipy import ndimage
import cv2

__all__ = [
    'getFloorplanPolygon',
    'resizeImage',
    'remakeImage',
    'openRgb',
    'openRgba'
]


def getFloorplanPolygon(img):
    if img.shape[2] > 3:
        img = Image.fromarray(img)
        imgwhite = Image.new("RGB", img.size, (255, 255, 255))
        imgwhite.paste(img, mask=img.split()[3])
        img = np.array(imgwhite)

    binary = np.all(img != [255, 255, 255], axis=2)
    binary = ndimage.binary_fill_holes(binary)

    poly = (s for i, (s, v) in enumerate(shapes(binary.astype(np.uint8), mask=binary)))
    results = []

    for l in poly:
        pts = np.array(l['coordinates'][0], dtype=np.int32).T
        
        results.append(pts[::-1])

    return sorted(results, key=lambda x: np.sum(x.mean(axis=1)))


def resizeImage(img, height=-1, width=-1):
    '''
    Mengganti ukuran gambar sesuai input.
    '''
    
    # tidak ada perubahan width dan height
    if width == -1 and height == -1:
        return img
    
    image_ratio = img.shape[0] / img.shape[1]
    
    # jika tidak ada perubahan salah satu ukuran
    # image rationya dipertahankan
    if height == -1:
        height = int(width * image_ratio)
    if width == -1:
        width = int(height / image_ratio)
    
    # proses resizing
    new_img = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    
    return new_img[:,:,::-1]


def remakeImage(img, bg, scale: tuple = (1,1), shift: tuple = (0,0), W: int = -1, H: int = -1):
    '''
    Menempelkan denah rumah pada background noise.
    Denah rumah memiliki 4 channel, channel ke-empat opacity
    
    Scale digunakan untuk mengatur ukuran gambar final,
    dengan faktor terhadap width dan height img.
    
    Shift mengatur seberapa jauh perpindahan img pada bg,
    respektif terhadap ukuran gambar.
    '''
    
    mask = img[:,:,3] > 0
    
    shape = img.shape
    new_shape = (int(shape[0] * scale[0]),
                 int(shape[1] * scale[1]))
    new_img = cv2.resize(bg, dsize=(new_shape[1], new_shape[0]), interpolation=cv2.INTER_CUBIC)
    
    loc = (int((new_shape[0] // 2 - shape[0] // 2) * (shift[0] + 1)),
           int((new_shape[1] // 2 - shape[1] // 2) * (shift[1] + 1)))
    
    new_img[loc[0] : loc[0] + shape[0],
            loc[1] : loc[1] + shape[1],
            : ][mask] = img[:,:,:3][mask]
    
    translation = np.array([
        [1, 0, loc[0]],
        [0, 1, loc[1]],
        [0, 0, 1]
    ])
    
    if W == -1 and H == -1:
        return new_img, translation
    
    resized_img = resizeImage(new_img, H, W)
    H = resized_img.shape[0]
    W = resized_img.shape[1]
    
    translation = np.dot(np.array([
        [H / new_shape[0], 0, 0],
        [0, W / new_shape[1], 0],
        [0, 0, 1]
    ]), translation)
    
    return resized_img[:,:,::-1], translation


def openRgb(path):
    img = Image.open(path)
    
    if len(img.getbands()) <= 3:
        return np.asarray(img)

    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    
    return np.asarray(background)


def openRgba(path):
    img = Image.open(path)
    
    return np.asarray(img)


def similarityFeature(img1, img2):
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