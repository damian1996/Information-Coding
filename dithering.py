import cv2 as cv
import numpy as np

def find_closest_palette_color(oldpix):
    return 255 * np.floor(oldpix/128)

image = cv.imread('lena.png', 0)
cv.imshow('dither', image)
img = image.copy()
sh = img.shape
print(sh)
for x in range(sh[0]):
    for y in range(sh[1]):
        oldpix = img[x,y]
        newpix = find_closest_palette_color(oldpix)
        img[x,y] = newpix
        quant_error = oldpix - newpix
        if x+1 < sh[0]:
            img[x+1,y] = img[x+1,y] + quant_error * (7/16)
        if x-1 >= 0 and y+1 < sh[1]:
            img[x-1,y+1] = img[x-1,y+1] + quant_error * (3/16)
        if y+1 < sh[1]:
            img[x,y+1] = img[x,y+1] + quant_error * (5/16)
        if x+1 < sh[0] and y+1 < sh[1]:
            img[x+1,y+1] = img[x+1,y+1] + quant_error * (1/16)

img = img.astype(np.uint8)

#im = Image.fromarray(np.uint8(img))
#im.show()
cv.imshow('dither_floyd_steinberg', img)

while True:
    k = cv.waitKey(0) & 0xFF     
    if k == 27: break 
cv.destroyAllWindows()