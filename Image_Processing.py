from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

image = cv2.imread('sample_image2.jpg')
## converting the image to Gray
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)
cv2.imwrite("grayscale_image.jpg",gray)

## threshold value to remove white background
ret,thresh_leaf = cv2.threshold(gray,215,255,cv2.THRESH_BINARY)
cv2.imshow("output_leaf",thresh_leaf)
cv2.waitKey(0)
cv2.imwrite("output_leaf.jpg",thresh_leaf)

#count leaf pixels
leaf_pix = cv2.countNonZero(thresh_leaf)
print(leaf_pix)

## threshold value to only see occluded stomata
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
ret,thresh_pseudo = cv2.threshold(gray,90,130,cv2.THRESH_BINARY_INV)
cv2.imshow("output_pseudo",thresh_pseudo)
cv2.waitKey(0)
cv2.imwrite("output_pseudo.jpg",thresh_pseudo)
cv2.destroyAllWindows()

#count pseudothecia pixels
pseudo_pix = cv2.countNonZero(thresh_pseudo)
print(pseudo_pix)

severity = pseudo_pix/leaf_pix * 100
print(severity)
