import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import cv2
import pickle
import os
import pims
from nd2reader import ND2Reader
from scipy import stats
import math

from __future__ import division
import matplotlib.image as mpimg
import scipy as sp
np.seterr(divide='ignore', invalid='ignore')


def GetSlice0(frames, numberslice): 
    c0 = frames.get_frame_2D(c=0, t=0, z=numberslice)
    return c0
def GetSlice1(frames, numberslice): 
    c1 = frames.get_frame_2D(c=1, t=0, z=numberslice)
    return c1

def ApplyThreshold(channel0, channel1, shape_image, thresholds):

    for i in range(0,shape_image[0]):
        for j in range(0,shape_image[1]):
            if channel0[i,j] <thresholds[0]:
                channel0[i,j]  =0
            else:
                channel0[i,j] =channel0[i,j]

    for i in range(0,shape_image[0]):
        for j in range(0,shape_image[1]):
            if channel1[i,j] <thresholds[1]:
                channel1[i,j]  =0
            else:
                channel1[i,j] =channel1[i,j]
    return   

def BlurThresh(channel0, channel1):
    img = channel0+channel1
    img = np.uint8(img)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,5)
    return img




def FindCircle(NPoints,Image, CorrCenterX,CorrCenterY,CorrectRadius):
    
    Circles = cv2.HoughCircles(Image, cv2.HOUGH_GRADIENT, dp=5, minDist=1, param2=10)
    n = (np.size(Circles))
    n = n/3
    n = int(n) 
    Circles = np.reshape(Circles, (n,3))
    d_theta = 2*np.pi/NPoints
    theta = np.arange(0, 2*np.pi, d_theta)
    CoordX = np.zeros( NPoints)
    CoordY = np.zeros( NPoints)
    
    Circle = np.array([Circles[0,0]+CorrCenterX, Circles[0,1]+CorrCenterY, Circles[0,2]+CorrectRadius])
    CoordX = Circle[0]+ Circle[2]*np.cos(theta)
    CoordY = Circle[1]+ Circle[2]*np.sin(theta)
    
    Coord = np.array([CoordX,CoordY])
    Coord = np.reshape(Coord.T, (NPoints,2))
    return Coord

def Plot (Image, Coordinates):
    plt.imshow(Image, cmap = 'gray', interpolation = 'bicubic')
    plt.plot(Coordinates[:,0],Coordinates[:,1], color = 'Red')
    plt.grid(True)
    plt.axis('equal')
    
def InfoCircle(NPoints,Image, CorrCenterX,CorrCenterY,CorrectRadius):
    
    Circles = cv2.HoughCircles(Image, cv2.HOUGH_GRADIENT, dp=5, minDist=1, param2=10)
    n = (np.size(Circles))
    n = n/3
    n = int(n) 
    Circles = np.reshape(Circles, (n,3))
    return Circles    
    
def CubeFit (NPoints,Image, alpha, CenterX, CenterY, Radius):
    
    d_theta = 2*np.pi/NPoints
    theta = np.arange(0, 2*np.pi, d_theta)
    comp1 = np.sqrt(np.abs(np.cos(theta)))*np.sign(np.cos(theta))
    comp2 = np.sqrt(np.abs(np.sin(theta)))*np.sign(np.sin(theta))
    CoordX = CenterX + Radius*(comp1*np.cos(alpha)-comp2*np.sin(alpha))
    CoordY = CenterY + Radius*(comp1*np.sin(alpha)+comp2*np.cos(alpha))
    Coord = np.array([CoordX,CoordY])
    Coord = np.reshape(Coord.T, (NPoints,2))
    return Coord  


def clopper_pearson(x, n, alpha=0.5):
    lo = beta.ppf(alpha / 2, x, n - x + 1)
    hi = beta.ppf(1 - alpha / 2, x + 1, n - x)
    return 0.0 if np.isnan(lo) else lo, 1.0 if np.isnan(hi) else hi  




def AreaFractionM (Channel0, Channel1, Coordinates, Npoints):
    
    CoordXOne= Coordinates[:,0]
    CoordYOne= Coordinates[:,1]
    CoordXOne= CoordXOne.astype(int)
    CoordYOne= CoordYOne.astype(int)
    I_0One = np.zeros( Npoints)
    I_1One = np.zeros( Npoints)
    I_0One = Channel0[CoordYOne, CoordXOne]
    I_1One = Channel1[CoordYOne, CoordXOne]
    I0_maxOne =np.amax(I_0One)
    I1_maxOne =np.amax(I_1One) 
    SpinsOne = np.zeros(Npoints)
    I_norm_0One = (I_0One)/I0_maxOne
    I_norm_1One = (I_1One)/I1_maxOne
    
    
    for i in range(Npoints):
    
        if I_norm_1One[i]==0 and I_norm_0One[i]>0.001 :
            SpinsOne[i] =-1
        elif I_norm_0One[i]==0 and I_norm_1One[i]>0.001 :
            SpinsOne[i] =1
        elif I_norm_0One[i]==0 and I_norm_1One[i]==0 :
            SpinsOne[i]==0
        elif I_norm_0One[i]>0.001 and I_norm_0One[i]>0.001  and I_norm_0One[i]>I_norm_1One[i]:
            SpinsOne[i]=-1
        elif I_norm_0One[i]>0.001 and I_norm_0One[i]>0.001  and I_norm_1One[i]>I_norm_0One[i]:
            SpinsOne[i]=1
        else:
            SpinsOne[i]=0        
        
    SpinsOne = SpinsOne.astype(int)  
    MaskOneOne = np.sum (SpinsOne==1)
    MaskMinusOneOne = np.sum (SpinsOne[:]==-1)
    MaskZeroOne = np.sum (SpinsOne[:]==0)
    AreaFractionMOne = 100*MaskOneOne/(Npoints-MaskZeroOne)
    clopper_low, clopper_high = np.asarray(clopper_pearson(MaskOneOne, Npoints, 0.5))
    s_clopper_err = clopper_low, clopper_high
    Error = (clopper_high-clopper_low)*100
    return np.array([AreaFractionMOne, Error])