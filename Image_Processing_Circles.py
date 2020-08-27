#import libraries
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import tkinter as tk
import csv

# script to change .tif images to png

# read in image
image = cv2.imread("Middle_2Y_D_Tip.png")

#color histogram
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    plt.ylim([0,10000])
plt.show()

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.imwrite("gray.jpg",gray)

#grayscale histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.ylim([0,1000])
plt.show()

##########################################################
########################### Needle ####################### 
##########################################################
edges = cv2.Canny(gray,100,150) #adjust
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
edges = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)
cv2.imshow("edged_gray",edges)
cv2.waitKey(0)

#needle_contour = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cv2.imshow("contoured_gray",needle_contour)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# or use simple thresholding with grayscale, all values close to white --> 0
ret,thresh_needle = cv2.threshold(gray,215,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("thresholding needle vs background",thresh_needle)
cv2.waitKey(0)
cv2.imwrite("thresh_needle.jpg",thresh_needle)

alpha = 0.5
beta = (1.0 - alpha)
dst = cv2.addWeighted(edges, alpha, thresh_needle, beta, 0.0)
cv2.imshow('dst', dst)
cv2.waitKey(0)
ret,re_thresh_needle = cv2.threshold(dst,1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("re-thresholding needle vs background",re_thresh_needle)
cv2.waitKey(0)

# Dilatation and erosion
kernel = np.ones((30,30), np.uint8)
img_dilation = cv2.dilate(re_thresh_needle, kernel, iterations=1)
img_erode = cv2.erode(img_dilation,kernel, iterations=1)
img_erode = cv2.medianBlur(img_erode, 7)
cv2.imshow('Dilatation + erosion',img_erode)
cv2.waitKey(0)
cv2.imwrite("dilatation_erosion.jpg",img_erode)

# remove noise outside needle using morph_open
kernel_noise = np.ones((5,5), np.uint8)
open_needle = cv2.morphologyEx(img_erode, cv2.MORPH_OPEN, kernel_noise)
cv2.imshow("needle without background noise",open_needle)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("open_needle.jpg",open_needle)

# uniformize needle using morph_close
kernel_needle = np.ones((10,10), np.uint8)
closed_needle = cv2.morphologyEx(open_needle, cv2.MORPH_CLOSE, kernel_needle)
cv2.imshow("needle without inside noise",closed_needle)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("closed_needle.jpg",closed_needle)

#count needle pixels
needle_pix = cv2.countNonZero(closed_needle)


#############################################################
######################## Pseudothecia #######################
#############################################################

# find black circles within leaf object contour (leaves only infected with SNC)
# all areas that are within the thresholds will go to 0 (background), everything else stays the same
#ret,thresh0_needle = cv2.threshold(gray, 200, 255, cv2.THRESH_TOZERO_INV | cv2.THRESH_OTSU)
#OR

#Make a stencil & mask out of the grayscale image & needle contour, removes noise, makes background white !

fill_color = [255, 255, 255] # any BGR color value to fill with
mask_value = 255            # 1 channel white (can be any non-zero uint8 value)

# our stencil - some `mask_value` contours on black (zeros) background, 
# the image has same height and width as `img`, but only 1 color channel
stencil  = np.zeros(image.shape[:-1]).astype(np.uint8)
contours, hierarchy = cv2.findContours(closed_needle,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.fillPoly(stencil, contours, mask_value)

sel = stencil != mask_value # select everything that is not mask_value
image[sel] = fill_color            # and fill it with fill_color
cv2.imshow("contoured needle", image)
cv2.waitKey(0)

# Create sharpening filter
filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# Apply cv2.filter2D function to sharpen image
sharpen_image =cv2.filter2D(image,-1,filter)
cv2.imshow("without noise, inside sharpened", sharpen_image)
cv2.waitKey(0)

#try using HSV
hsv = cv2.cvtColor(sharpen_image, cv2.COLOR_BGR2HSV)
ret, thresh_pseudo = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("output_pseudo_hsv",thresh_pseudo)
cv2.waitKey(0)

#HSV not working, try grayscale
sharpen_gray = cv2.cvtColor(sharpen_image, cv2.COLOR_RGB2GRAY)
ret,thresh_pseudo = cv2.threshold(sharpen_gray,50,255,cv2.THRESH_BINARY_INV)
cv2.imshow("output_pseudo",thresh_pseudo)
cv2.waitKey(0)
cv2.imwrite("output_pseudo.jpg",thresh_pseudo)

# Dilatation and erosion
kernel = np.ones((2,2), np.uint8)
pseudo_dilation = cv2.dilate(thresh_pseudo, kernel, iterations=1)
pseudo_erode = cv2.erode(pseudo_dilation,kernel, iterations=1)
pseudo_erode = cv2.medianBlur(pseudo_erode, 7)
cv2.imshow('Dilatation + erosion',pseudo_erode)
cv2.waitKey(0)
cv2.destroyAllWindows()


# remove noise within circles
# find circles
edges_pseudo = cv2.Canny(gray,0,50) #adjust
cv2.imshow("Pseudothecia edges",edges_pseudo)
cv2.waitKey(0)
kernel_circles = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
morph = cv2.morphologyEx(pseudo_erode, cv2.MORPH_CLOSE, kernel_circles)
dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
cv2.imshow("Dist Image",dist)
cv2.waitKey(0)
cv2.imwrite("dist_image.jpg",thresh_pseudo)
cv2.destroyAllWindows()

# a 10 pixel template to match with the image
borderSize = 10
distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
gap = 9                                
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
distTempl = cv2.distanceTransform(kernel2, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
cv2.imshow("Dist * Template Image",distTempl)
cv2.waitKey(0)
cv2.imwrite("dist_template_image.jpg",thresh_pseudo)
cv2.destroyAllWindows()

nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
mn, mx, _, _ = cv2.minMaxLoc(nxcor)
th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
peaks8u = cv2.convertScaleAbs(peaks)
contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
    cv2.circle(im, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.drawContours(im, contours, i, (0, 0, 255), 2)

cv2.imshow('circles', im)
## ^ above from https://stackoverflow.com/questions/26932891/detect-touching-overlapping-circles-ellipses-with-opencv-and-python

# count number of pixels for each black circular object
# translate this to proportion of leaf occupied by each black circular object
# histogram of those proportions + size threshold to discriminate
# count pixels of black circular objects below the threshold
# this gives us the proportion of leaf occupied by pseudothecia !
# output value to a xlsx
