__author__ = 'p2admin'

import os.path
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

address = ''
filename = 'image.png'
img = cv2.imread(address + filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, None, fx=0.5, fy=0.5, )
row_, col_ = gray.shape

# canny edge detection
v = np.median(gray)
sigma = 0.50

lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

edges = cv2.Canny(img,threshold1 = lower, threshold2 = upper)
plt.subplot(121),plt.imshow(gray,'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,'gray'),plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()