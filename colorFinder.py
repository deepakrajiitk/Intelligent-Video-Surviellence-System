import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import colorsys
from collections import Counter
from sklearn.mixture import GaussianMixture as GMM
clt_gmm = GMM(n_components=1)

def rgb_to_hsv(c):
    c = np.array(c)/255
    c  = colorsys.rgb_to_hsv(c[0],c[1],c[2])
    return c

def palette_perc(k_cluster, img):
    # width = 300
    # palette = np.zeros((50, width, 3), np.uint8)
    
    clusters = k_cluster.predict(img.reshape(-1, 3))
    n_pixels = len(clusters)
    counter = Counter(clusters) # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i]/n_pixels, 2)
    perc = dict(sorted(perc.items()))
    
    # print(perc)
    clr = k_cluster.means_.astype(np.uint16)
    Key_max = max(perc, key = lambda x: perc[x])  
    # print(clr)
    return clr[Key_max]

def hueRange(hue):
    if hue>=0 and hue<35:
        return "red", 0
    elif hue>=35 and hue<80:
        return "yellow", 1
    elif hue>=80 and hue<170:
        return "green", 2
    elif hue>=170 and hue<200:
        return "cyan", 3
    elif hue>=200 and hue<270:
        return "blue", 4
    elif hue>=270 and hue<=360:
        return "pink", 5

def findColor(img):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clt_1 = clt_gmm.fit(img.reshape(-1, 3))
    clr = palette_perc(clt_1, img)
    # show_img_compar(img, pallete)
    # print(clr)
    hsvClr = rgb_to_hsv(clr)

    # print(hsvClr)
    if hsvClr[2]<=0.1:
        idx = 7
        hue = "black"
    elif hsvClr[1]<=0.1:
        if hsvClr[2]>=0.5:
            idx = 6
            hue = "white"
        else:
            idx=8
            hue ="gray"
    else:
        hue = hsvClr[0]*360
        hue , idx= hueRange(hue)
    
    return hue, idx

