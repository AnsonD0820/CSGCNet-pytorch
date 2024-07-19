
import numpy as np
from PIL import Image

img = Image.open("000001_left2.png").convert('P')
colormap = np.array([
    [  0,  0,  0],
    [102 ,102 ,156],
    [128, 64 ,128],
    [107 ,142, 35],
    [  0,  0 ,142],
    [ 70, 70, 70]
] ,dtype=np.uint8)
img.putpalette(colormap.flatten())
img.save("test.png")
print(img)