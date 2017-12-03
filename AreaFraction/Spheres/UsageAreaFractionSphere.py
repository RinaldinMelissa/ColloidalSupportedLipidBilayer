C0 = GetSlice1(slide, 0) #slice zero
C1 = GetSlice0(slide, 0)
ApplyThreshold(C0, C1, np.array(np.shape(C0)), np.array([100,400])) # thresholds for the two channels
Img = BlurThresh(C0,C1)
Coordinates =FindCircle(1000,Img, 0,0,0)  #correction for a bettter fit (here 0,0,0)
Plot(Img, Coordinates )
AreaFractionM (C0, C1, Coordinates, 1000)