import matplotlib.pylab as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters
import numpy as np
from numba import jit

def getSeam(imgray, e):
    inf = 100000
    #energy map generation
    M = np.empty(e.shape)

    M[1] = e[0]
    M[1,0] = M[0,-1] = inf

    for i in range(2, len(e)-1):
        for j in range(1,len(e[0])-1): #range correction: note e dimensions are (n+2)*(m+2) while image dimensions are n*m
            M[i,j] = e[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])
            M[i,0] = M[i,-1] = inf

    #backtracking
    seam = np.empty(len(imgray), dtype = 'int')
    
    minEnergyPixel = np.argmin(M[len(imgray)-1])
    seam[-1] = minEnergyPixel

    M = np.delete(M, (0,len(M)-1), axis = 0)
    M = np.delete(M, (0,len(M[0]-1)), axis = 1)

    for i in reversed(range(len(imgray)-1)):
        if minEnergyPixel > 1 and minEnergyPixel < len(imgray)-1:
            minEnergyPixel += np.argmin(M[i,minEnergyPixel-1:minEnergyPixel+2]) - 1
        elif minEnergyPixel > 1:
            minEnergyPixel += np.argmin(M[i,minEnergyPixel-1:minEnergyPixel+1]) - 1
        else:
            minEnergyPixel += np.argmin(M[i,minEnergyPixel:minEnergyPixel+2])

        seam[i] = minEnergyPixel

    return seam

im = imread('Broadway_tower_edit.jpg') # RGB image to gray scale
imgray = rgb2gray(im)

#Add borders to imgray to avoid losing pixels due to 0's appearing in borders due to Sobel
expandedImgray = np.vstack((imgray[0], imgray, imgray[-1]))
nrows = expandedImgray.shape[0]
expandedImgray = np.hstack((expandedImgray[:,0].reshape(nrows,1), expandedImgray, expandedImgray[:,-1].reshape(nrows,1)))

e = filters.sobel(expandedImgray)

seam = getSeam(imgray, e)
print(seam)

plt.subplot(1,3,1)
plt.imshow(e, cmap = 'gray')
plt.plot(seam, list(range(len(seam))), linewidth = 0.5, color = 'r')

e = np.delete(e, (0,len(e)-1), axis = 0)
e = np.delete(e, (0,len(e[0])-1), axis = 1)


plt.show()