from PIL import Image
import io
import numpy as np
import skimage

__all__ = [
    'qualityReduction',
    'addNoise'
]

def qualityReduction(img, q=20):
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
        img = (img / 255.0).astype(np.float32)
    
    noised = skimage.util.random_noise(img, mode=mode)
    noised = (noised * 255.0).astype(np.uint8)
    return noised