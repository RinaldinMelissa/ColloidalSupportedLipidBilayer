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




def FindCircles(NPoints,Image, CorrCenterXOne,CorrCenterYOne,CorrectRadiusOne, CorrCenterXTwo,CorrCenterYTwo,CorrectRadiusTwo):
    
    Circles = cv2.HoughCircles(Image, cv2.HOUGH_GRADIENT, dp=5, minDist=1, param2=10)
    n = (np.size(Circles))
    n = n/3
    n = int(n) 
    Circles = np.reshape(Circles, (n,3))
    d_theta = 2*np.pi/NPoints
    theta = np.arange(0, 2*np.pi, d_theta)
    CoordX_One = np.zeros( NPoints)
    CoordY_One = np.zeros( NPoints)
    CoordX_Two = np.zeros( NPoints)
    CoordY_Two = np.zeros( NPoints)
    CircleOne = np.array([Circles[0,0]+CorrCenterXOne, Circles[0,1]+CorrCenterYOne, Circles[0,2]+CorrectRadiusOne])
    CircleTwo = np.array([Circles[1,0]+CorrCenterXTwo, Circles[1,1]+CorrCenterYTwo, Circles[1,2]+CorrectRadiusTwo])
    CoordX_One = CircleOne[0]+ CircleOne[2]*np.cos(theta)
    CoordY_One = CircleOne[1]+ CircleOne[2]*np.sin(theta)
    CoordX_Two = CircleTwo[0]+ CircleTwo[2]*np.cos(theta)
    CoordY_Two = CircleTwo[1]+ CircleTwo[2]*np.sin(theta)
    Coord = np.array([CoordX_One,CoordY_One, CoordX_Two, CoordY_Two])
    Coord = np.reshape(Coord.T, (NPoints,4))
    return Coord

def Plot (Image, Coordinates):
    plt.imshow(Image, cmap = 'gray', interpolation = 'bicubic')
    plt.plot(Coordinates[:,0],Coordinates[:,1], color = 'Red')
    plt.plot(Coordinates[:,2],Coordinates[:,3], color = 'Blue')
    plt.grid(True)
    plt.axis('equal')
    
    
    
    
    
def AreaFractionM (Channel0, Channel1, Coordinates, Npoints):
    
    CoordXOne= Coordinates[:,0]
    CoordYOne= Coordinates[:,1]
    CoordXTwo= Coordinates[:,2]
    CoordYTwo= Coordinates[:,3]
    
    CoordXOne= CoordXOne.astype(int)
    CoordYOne= CoordYOne.astype(int)
    CoordXTwo= CoordXTwo.astype(int)
    CoordYTwo= CoordYTwo.astype(int)
    
    I_0One = np.zeros( Npoints)
    I_1One = np.zeros( Npoints)
    I_0Two = np.zeros( Npoints)
    I_1Two = np.zeros( Npoints)
    
    
    
    I_0One = Channel0[CoordYOne, CoordXOne]
    I_1One = Channel1[CoordYOne, CoordXOne]
    I_0Two = Channel0[CoordYTwo, CoordXTwo]
    I_1Two = Channel1[CoordYTwo, CoordXTwo]
    
    
    I0_maxOne =np.amax(I_0One)
    I1_maxOne =np.amax(I_1One)
    I0_maxTwo =np.amax(I_0Two)
    I1_maxTwo =np.amax(I_1Two)
    
    
    SpinsOne = np.zeros(Npoints)
    SpinsTwo = np.zeros(Npoints)

    I_norm_0One = (I_0One)/I0_maxOne
    I_norm_1One = (I_1One)/I1_maxOne
    I_norm_0Two = (I_0Two)/I0_maxTwo
    I_norm_1Two = (I_1Two)/I1_maxTwo
    
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
            
            
            
    for i in range(Npoints):
    
        if I_norm_1Two[i]==0 and I_norm_0Two[i]>0.001 :
            SpinsTwo[i] =-1
        elif I_norm_0Two[i]==0 and I_norm_1Two[i]>0.001 :
            SpinsTwo[i] =1
        elif I_norm_0Two[i]==0 and I_norm_1Two[i]==0 :
            SpinsTwo[i]==0
        elif I_norm_0Two[i]>0.001 and I_norm_0Two[i]>0.001  and I_norm_0Two[i]>I_norm_1Two[i]:
            SpinsTwo[i]=-1
        elif I_norm_0Two[i]>0.001 and I_norm_0Two[i]>0.001  and I_norm_1Two[i]>I_norm_0Two[i]:
            SpinsTwo[i]=1
        else:
            SpinsTwo[i]=0
            
        
    SpinsOne = SpinsOne.astype(int)  
    MaskOneOne = np.sum (SpinsOne==1)
    MaskMinusOneOne = np.sum (SpinsOne[:]==-1)
    MaskZeroOne = np.sum (SpinsOne[:]==0)
    SpinsTwo = SpinsTwo.astype(int)  
    MaskOneTwo = np.sum (SpinsTwo==1)
    MaskMinusOneTwo = np.sum (SpinsTwo[:]==-1)
    MaskZeroTwo = np.sum (SpinsTwo[:]==0)
    

    AreaFractionMOne = 100*MaskOneOne/(Npoints-MaskZeroOne)
    AreaFractionMTwo = 100*MaskOneTwo/(Npoints-MaskZeroTwo)
    
    AreaFractionM = (AreaFractionMOne+AreaFractionMTwo)/2
    return AreaFractionM