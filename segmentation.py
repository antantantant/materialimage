__author__ = 'p2admin'
import sys
import cv2
import numpy
from scipy.ndimage import label
from matplotlib import pyplot as plt

def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255/ncc)
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl


address = ''
filename = 'image.png'
img = cv2.imread(address + filename)


# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img_bin = cv2.threshold(img_gray, 0, 255,
        cv2.THRESH_OTSU)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=int))
plt.subplot(1,3,1)
plt.imshow(img_bin),plt.title('original'),plt.xticks([]), plt.yticks([])


result = segment_on_dt(img, img_bin)
plt.subplot(1,3,2)
plt.imshow(result),plt.title('segment'),plt.xticks([]), plt.yticks([])

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)

plt.subplot(1,3,3)
plt.imshow(result),plt.title('dilate'),plt.xticks([]), plt.yticks([])
plt.show()