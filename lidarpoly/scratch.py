from PIL import Image
import numpy as np

img = Image.open("/home/rsulzer/data/test_image.tif")
print(img.size)

np_img = np.array(img)
print(np_img.shape)

a=5