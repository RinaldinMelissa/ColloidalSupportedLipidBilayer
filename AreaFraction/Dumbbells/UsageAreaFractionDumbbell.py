C0 = GetSlice1(slide,0) #get slice 0
C1 = GetSlice0(slide,0)
ApplyThreshold(C0, C1, np.array(np.shape(C0)), np.array([0,50])) #apply threshold
Img = BlurThresh(C0,C1)
CoordinatesNephroid = FindNephroid(1000, 30,40,9,-0.3) #fit a nephroid
Plot(Img,CoordinatesNephroid[:,0], CoordinatesNephroid[:,1] )
AreaFractionM(C0, C1, CoordinatesNephroid[:,0], CoordinatesNephroid[:,1], 1000)