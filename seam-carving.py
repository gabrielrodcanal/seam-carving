# Copyright (C) 2018 Gabriel Rodríguez Canal
# 
# This file is part of Seam carving.
# 
# Seam carving is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Seam carving is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Seam carving.  If not, see <http://www.gnu.org/licenses/>.



import matplotlib.pylab as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters
import numpy as np
from numba import jit


nSeams = 200
enlarging = True

@jit
def getSeam(imgray, e):
    inf = 100000
    #energy map generation
    M = np.empty(e.shape)

    M[1] = e[1]
    M[1,0] = M[1,-1] = inf

    for i in range(2, len(e)-1):
        for j in range(1,len(e[0])-1): #range correction: note e dimensions are (n+2)*(m+2) while image dimensions are n*m
            M[i,j] = e[i,j] + min(M[i-1,j-1], M[i-1,j], M[i-1,j+1])
            M[i,0] = M[i,-1] = inf

    #backtracking
    seam = np.empty(len(imgray), dtype = 'int')
    
    M = np.delete(M, (0,len(M)-1), axis = 0)

    minEnergyPixel = np.argmin(M[len(imgray)-1])
    seam[-1] = minEnergyPixel    

    for i in reversed(range(len(imgray)-1)):
        #infinite values in edges enable avoiding border checking
        minEnergyPixel += np.argmin(M[i,minEnergyPixel-1:minEnergyPixel+2]) - 1

        seam[i] = minEnergyPixel-1

    return seam

#Seam coordinates must be local to imRGB
@jit
def removeSeam(imRGB, seam):
    imRGBSeamRemoved = np.empty((len(imRGB), len(imRGB[0])-1, 3), dtype = 'uint8')

    for row in range(len(seam)):
        imRGBSeamRemoved[row] = np.concatenate((imRGB[row, :seam[row]], imRGB[row, seam[row]+1:]))

    return imRGBSeamRemoved

def addSeam(imRGB, seam):
    imRGBSeamEnlarged = np.empty((len(imRGB), len(imRGB[0])+1, 3), dtype = 'uint8')

    for row in range(len(seam)):
        #new seam value should be averaged with neighbours
        if seam[row] > 1 and seam[row] < len(imRGB[0])-1:
            newValue = np.mean(imRGB[row, seam[row]-1:seam[row]+2], axis = 0)
        elif seam[row] > 1:
            newValue = np.mean(imRGB[row, seam[row]-1:seam[row]+1], axis = 0)
        else:
            newValue = np.mean(imRGB[row, seam[row]:seam[row]+2], axis = 0)
        imRGBSeamEnlarged[row] = np.concatenate((imRGB[row, :seam[row]], [newValue],  imRGB[row, seam[row]:]))

    return imRGBSeamEnlarged


#Transform last seam local coordinates into global coordinates
@jit
def fixLastSeam(seams):
    if(len(seams) > 1):
        previousSeams = seams[:len(seams)-1]
        lastSeam = seams[len(seams)-1]

        for row in range(len(previousSeams[0])):
            lastSeam[row] += np.count_nonzero(previousSeams[:,row] < lastSeam[row])

def fixNextSeam(currentSeamIndex, seams):
    if(len(seams) > 1):
        currentSeam = seam[currentSeamIndex]
        otherSeams = np.concatenate((seam[:currentSeamIndex], seam[currentSeamIndex+1:]))

    for row in range(len(otherSeams[0])):
        #seams are shifted by one pixel if current seam is located before them, since it will be duplicated
        otherSeams[:,row] = list(map(lambda x: x+1 if x > currentSeam[row] else x, otherSeams[:,row]))

    return np.concatenate((otherSeams[:currentSeamIndex], [currentSeam], otherSeams[currentSeamIndex:]))


imRGBOriginal = imread('Broadway_tower_edit.jpg') # RGB image to gray scale
imRGBSeamCarved = np.copy(imRGBOriginal)
seam = np.empty((nSeams, len(imRGBOriginal)), dtype = 'int')
eOriginal = filters.sobel(rgb2gray(imRGBOriginal))


for s in range(nSeams):
    print('Processing seam ', s)
    imgray = rgb2gray(imRGBSeamCarved)

    #Add borders to imgray to avoid losing pixels due to 0's appearing in borders due to Sobel
    expandedImgray = np.vstack((imgray[0], imgray, imgray[-1]))
    nrows = expandedImgray.shape[0]
    expandedImgray = np.hstack((expandedImgray[:,0].reshape(nrows,1), expandedImgray, expandedImgray[:,-1].reshape(nrows,1)))

    e = filters.sobel(expandedImgray)

    seam[s] = getSeam(imgray, e)

    imRGBSeamCarved = removeSeam(imRGBSeamCarved, seam[s])
    
    fixLastSeam(seam)

if enlarging:
    imRGBSeamCarved = np.copy(imRGBOriginal)
    for s in range(nSeams):
        imRGBSeamCarved = addSeam(imRGBSeamCarved, seam[s])
        seam = fixNextSeam(s, seam)

    eOriginal = filters.sobel(rgb2gray(imRGBSeamCarved))


plt.subplot(1,3,1)
plt.title('Energy map with seams')
plt.imshow(eOriginal, cmap = 'gray')

for s in range(nSeams):
    plt.plot(seam[s], list(range(len(seam[0]))), linewidth = 0.5, color = 'r')

e = np.delete(e, (0,len(e)-1), axis = 0)
e = np.delete(e, (0,len(e[0])-1), axis = 1)

plt.subplot(1,3,2)
plt.title('Original image')
plt.imshow(imRGBOriginal)

plt.subplot(1,3,3)
plt.title('Seam carved image')
plt.imshow(imRGBSeamCarved)

plt.show()