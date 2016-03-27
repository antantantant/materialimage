__author__ = 'yiren'
import os.path
import pickle
import cv2
import numpy as np
from scipy.ndimage import label
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max

# for image preprocessing
import lic_internal
import FDoGedge

# for regression with bspline
from uniform_bspline import UniformBSpline
from fit_uniform_bspline import UniformBSplineLeastSquaresOptimiser
from scipy.spatial.distance import cdist

# for clustering
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import networkx as nx

# get normalized gradient from sobel filter
def sobel(img):
    gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grady = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.dstack((gradx, grady))
    grad_norm = np.linalg.norm(grad, axis=2)
    st = np.dstack((gradx*gradx, grady*grady, gradx*grady))
    return grad, grad_norm, st

# main function to update tangent
# ita, r: wm parameter, neighbourhood radius
def edge_tangent_flow(img):
    # step 1: Calculate the struture tensor
    grad, grad_norm, st = sobel(img)
    row_, col_ = img.shape

    # step 2: Gaussian blur the struct tensor. sst_sigma = 2.0
    sigma_sst = 2.0
    gaussian_size = int((sigma_sst*2)*2+1)
    blur = cv2.GaussianBlur(st, (gaussian_size,gaussian_size), sigma_sst)

    tan_ETF = np.zeros((row_,col_,2))
    E = blur[:,:,0]
    G = blur[:,:,1]
    F = blur[:,:,2]

    lambda2 = 0.5*(E+G-np.sqrt((G-E)*(G-E)+4.0*F*F))
    v2x = (lambda2 - G != 0) * (lambda2 - G) + (lambda2 - G == 0) * F
    v2y = (lambda2 - G != 0) * F + (lambda2 - G == 0) * (lambda2 -E)
    # v2x = cv2.GaussianBlur(v2x, (gaussian_size,gaussian_size), sigma_sst)
    # v2y = cv2.GaussianBlur(v2y, (gaussian_size,gaussian_size), sigma_sst)
    v2 = np.sqrt(v2x*v2x+v2y*v2y)
    tan_ETF[:,:,0] = v2x/(v2+0.0000001)*((v2!=0)+0)
    tan_ETF[:,:,1] = v2y/(v2+0.0000001)*((v2!=0)+0)

    # plt.subplot(1,3,1)
    # plt.imshow(tan_ETF[:,:,0],'gray')
    # plt.subplot(1,3,2)
    # plt.imshow(tan_ETF[:,:,1],'gray')
    return tan_ETF

# Visualize a vector field by using LIC (Linear Integral Convolution).
def visualizeByLIC(vf):
    row_,col_,dep_ = vf.shape
    texture = np.random.rand(col_,row_).astype(np.float32)
    kernellen=9
    kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
    kernel = kernel.astype(np.float32)
    vf = vf.astype(np.float32)
    img = lic_internal.line_integral_convolution(vf, texture, kernel)
    return img

def segment_on_dt(a, img, gray):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)


    dt = cv2.distanceTransform(img,cv2.DIST_L2,5)
    plt.subplot(3,3,4)
    plt.imshow(dt),plt.title('dt'),plt.xticks([]), plt.yticks([])

    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt2 = cv2.threshold(dt, 0, 255, cv2.THRESH_BINARY)
    dt2 = cv2.erode(dt2, None, iterations=2)
    # dt2 = cv2.adaptiveThreshold(dt, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    # dt1 = peak_local_max(gray, indices=False, min_distance=10, labels=img, threshold_abs=5)
    # dt2 = peak_local_max(dt, indices=False, min_distance=5, labels=img, threshold_abs=0)
    lbl, ncc = label(dt2)

    plt.subplot(3,3,5)
    plt.imshow(dt2),plt.title('localMax'),plt.xticks([]), plt.yticks([])
    # plt.subplot(3,3,6)
    # plt.imshow(ncc),plt.title('ncc'),plt.xticks([]), plt.yticks([])

    lbl = lbl * (255/ncc)
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)

    plt.subplot(3,3,6)
    plt.imshow(lbl),plt.title('lbl_out'),plt.xticks([]), plt.yticks([])
    return 255 - lbl

#######################################################################################################
#################################### main code ########################################################
#######################################################################################################
address = ''
filename = 'image.png'
img = cv2.imread(address + filename)
img = cv2.resize(img, None, fx=0.5, fy=0.5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
row_, col_ = gray.shape
plt.subplot(3,3,1)
plt.imshow(gray,'gray')

debug = False
############################## The following section preprocess the image ##############################
fdog_img_address = 'fdog_test_data.out'
etf_address = 'etf_test_data.out'

#fdog parameters
sigma_e = 1.0
sigma_r = 1.6 #1.6
sigma_m = 4.0 #2.0
tau = 0.99 #0.99
phi = 2.0 #2.0
threshold = 0.5 #0.5
fdog_loop = 6

if os.path.isfile(fdog_img_address+'.npy') and os.path.isfile(etf_address+'.npy') and not debug:
    fdog_img = np.load(fdog_img_address+'.npy')
    tan_ETF = np.load(etf_address+'.npy')
else:
    ## shrink the image to a pre-defined standard
    gray_ = gray.copy()

    ## Calculate tangent flow
    tan_ETF = edge_tangent_flow(gray_)

    ## FDoG loop
    for count in np.arange(fdog_loop):
        ## Get FDoG Edge
        fdog_img, f0, f1 = FDoGedge.getFDoGedge(tan_ETF.astype(np.float64),gray_.astype(np.float64),
                                        sigma_e,sigma_r,sigma_m,tau,phi,threshold)
        gray_[fdog_img<255]=0.0
        # tan_ETF = edge_tangent_flow(gray_)
    np.save(fdog_img_address, fdog_img)
    np.save(etf_address, tan_ETF)

lic_img = visualizeByLIC(tan_ETF)
plt.subplot(3,3,2)
plt.imshow(lic_img,'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3)
plt.imshow(fdog_img,'gray'),plt.title('Result: sigma_e='+ str(sigma_e) + ', sigma_r='+ str(sigma_r)
                                      + ', sigma_m=' + str(sigma_m) + ', tau=' + str(tau)
                                      + ', phi=' + str(phi) + ', threshold=' + str(threshold)
                                      + ', fdog_loop=' + str(fdog_loop)),plt.xticks([]), plt.yticks([])

############ Segmentation.
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, img_bin = cv2.threshold(img_gray, 0, 255,
#         cv2.THRESH_OTSU)
# fdog_img = cv2.morphologyEx(fdog_img, cv2.MORPH_OPEN,
#         np.ones((2, 2), dtype=int), iterations=3)
# plt.subplot(2,3,4)
# plt.imshow(fdog_img),plt.title('opening'),plt.xticks([]), plt.yticks([])

result = segment_on_dt(img.astype('uint8'), fdog_img.astype('uint8'), gray)
plt.subplot(3,3,7)
plt.imshow(result),plt.title('segment'),plt.xticks([]), plt.yticks([])

result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)

plt.subplot(3,3,8)
plt.imshow(result),plt.title('dilate'),plt.xticks([]), plt.yticks([])

plt.subplot(3,3,9)
plt.imshow(img),plt.title('result'),plt.xticks([]), plt.yticks([])

segmentation_address = 'segmentation.out'
np.save(segmentation_address, img)

plt.show()


