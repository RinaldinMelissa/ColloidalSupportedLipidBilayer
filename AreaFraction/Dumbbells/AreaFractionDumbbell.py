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

def FindNephroid(NPoints,CenterX, CenterY, Radius, Alpha) :
    d_theta = 2*np.pi/NPoints
    theta = np.arange(0, 2*np.pi, d_theta)
    x = np.zeros( NPoints)
    y = np.zeros( NPoints)
    comp1  = 6*Radius*np.cos(theta) -4*Radius*(np.cos(theta))**3
    comp2 = 4*Radius*(np.sin(theta))**3
    x = CenterX + (comp1*np.cos(Alpha)-comp2*np.sin(Alpha))
    y = CenterY + (comp1*np.sin(Alpha)+comp2*np.cos(Alpha))
    coord = np.array([x,y])
    coord = np.reshape(coord.T, (1000,2))
    return coord
    

def Plot(Image, CoordX, CoordY):
    plt.imshow(Image, cmap = 'gray', interpolation = 'bicubic')
    plt.plot(CoordX,CoordY)
    plt.grid(True)
    plt.axis('equal')
def AreaFractionM (Channel0, Channel1, CoordX, CoordY, Npoints):
    CoordX= CoordX.astype(int)
    CoordY= CoordY.astype(int)
    I_0 = np.zeros( Npoints)
    I_1 = np.zeros( Npoints)
    I_0 = Channel0[CoordY, CoordX]
    I_1 = Channel1[CoordY, CoordX]
    I0_max =np.amax(I_0)
    I1_max =np.amax(I_1)
    Spins = np.zeros(Npoints)
    I_norm_0 = (I_0)/I0_max 
    I_norm_1 = (I_1)/I1_max
    
    
    for i in range(Npoints):
    
        if I_norm_1[i]==0 and I_norm_0[i]>0.001 :
            Spins[i] =-1
        elif I_norm_0[i]==0 and I_norm_1[i]>0.001 :
            Spins[i] =1
        elif I_norm_0[i]==0 and I_norm_1[i]==0 :
            Spins[i]==0
        elif I_norm_0[i]>0.001 and I_norm_0[i]>0.001  and I_norm_0[i]>I_norm_1[i]:
            Spins[i]=-1
        elif I_norm_0[i]>0.001 and I_norm_0[i]>0.001  and I_norm_1[i]>I_norm_0[i]:
            Spins[i]=1
        else:
            print('error')
    Spins = Spins.astype(int)  
    MaskOne = np.sum (Spins==1)
    MaskMinusOne = np.sum (Spins[:]==-1)
    MaskZero = np.sum (Spins[:]==0)
    AreaFractionM = 100*MaskOne/(Npoints-MaskZero)
    return AreaFractionM
   