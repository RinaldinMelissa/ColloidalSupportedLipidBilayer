C0 = GetSlice1(slide,0) #choose slice zero
C1 = GetSlice0(slide,0)
ApplyThreshold(C0, C1, np.array(np.shape(C0)), np.array([100,100])) #threshold
Img = BlurThresh(C0,C1)
Coordinates =FindCircles(1000,Img, 0,0,4,0,0,4) #inerpolate with two circles
Plot(Img, Coordinates )
AreaFractionM (C0, C1, Coordinates, 1000)