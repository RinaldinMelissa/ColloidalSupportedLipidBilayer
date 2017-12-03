C0 = GetSlice1(slide,0) #choose slice zero
C1 = GetSlice0(slide,0)
ApplyThreshold(C0, C1, np.array(np.shape(C0)), np.array([100,400])) #threshold bounds
Img = BlurThresh(C0,C1)
Coordinates = FindCircle(1000,Img, 0,0,0) #first find the circle that interpolates the squircle
Info =InfoCircle(1000,Img, 0,0,0)
NewCoordinates = CubeFit (1000,Img, 1, Info[0,0], Info[0,1], Info[0,2]) #the interpolate with a squircle curve
Plot(Img, NewCoordinates )
AreaFractionM (C0, C1, NewCoordinates, 1000)