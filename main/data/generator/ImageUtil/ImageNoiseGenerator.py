from PIL import Image
import io, cv2
import numpy as np
import skimage

__all__ = [
    'qualityReduction',
    'addNoise',
    'modifyHue'
]

def qualityReduction(img, q=60):
    with io.BytesIO() as output_buffer:
        size = img.shape
        img = Image.fromarray(img)
        img.save(output_buffer, optimize=True, quality=q, format="JPEG")
        
        compressed_image = Image.open(output_buffer)
        
        return np.array(compressed_image)
    return img

def addNoise(img, mode):
    if mode is None:
        return img
    
    if isinstance(img, np.ndarray) and issubclass(type(img.dtype), np.integer):
        img = (img / 127.0).astype(np.float32)
    
    noised = skimage.util.random_noise(img, mode=mode)
    noised = (noised * 255.0).astype(np.uint8)
    return noised

def modifyHue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = (hsv[:,:,0] + np.random.random_integers(low=-50, high=50)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)