from seamcarving import *
import sys
from PIL import Image

def console():
    print('Enter the image source path:')
    path = input('> ')

    imRGBOriginal = imread(path) # RGB image to gray scale

    print('Which orientation do you wish, horizontal (H) or vertical (V)?')
    orientation = ''
    orientation = input('> ')
    while orientation not in ['H', 'h', 'V', 'v']:
        print('Wrong orientation.')
        print('Which orientation do you wish, horizontal (H) or vertical (V)?')
        orientation = input('> ')

    print('Would you like to reduce (R) or to enlarge (E) the image?')
    sizeOption = ''
    sizeOption = input('> ')
    while sizeOption not in ['R', 'r', 'E', 'e']:
        print('Wrong orientation.')
        print('Which orientation do you wish, horizontal (H) or vertical (V)?')
        sizeOption = input('> ')

    print('How many seams do you wish to calculate?')
    nSeams = int(input('> '))

    if orientation in ['V', 'v']: horizontal = False
    else: horizontal = True

    if sizeOption in ['R', 'r']: enlargement = False
    else: enlargement = True

    seamCarve(imRGBOriginal, horizontal, enlargement, nSeams)

@jit
def seamCarve(imRGBOriginal, horizontal, enlargement, nSeams):
    if horizontal:
        imRGBOriginal = imRGBOriginal.transpose((1,0,2))

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

    if enlargement:
        imRGBSeamCarved = np.copy(imRGBOriginal)
        for s in range(nSeams):
            imRGBSeamCarved = addSeam(imRGBSeamCarved, seam[s])
            seam = fixNextSeam(s, seam)

        eOriginal = filters.sobel(rgb2gray(imRGBSeamCarved))

    if horizontal:
        imRGBOriginal = imRGBOriginal.transpose((1,0,2))
        eOriginal = eOriginal.transpose()
        imRGBSeamCarved = imRGBSeamCarved.transpose((1,0,2))

    fig = plt.figure()

    plot1 = fig.add_subplot(1,3,1)
    plot1.imshow(eOriginal, cmap = 'gray')

    if horizontal:
        for s in range(nSeams):
            plot1.plot(list(range(len(seam[0]))), seam[s], linewidth = 0.5, color = 'r')

    else:
        for s in range(nSeams):
            plot1.plot(seam[s], list(range(len(seam[0]))), linewidth = 0.5, color = 'r')

    e = np.delete(e, (0,len(e)-1), axis = 0)
    e = np.delete(e, (0,len(e[0])-1), axis = 1)

    if not horizontal:
        plot2 = fig.add_subplot(1,3,2, sharey = plot1)
    else:
        plot2 = fig.add_subplot(1,3,2, sharex = plot1)

    plot2.imshow(imRGBOriginal)

    if not horizontal:
        plot3 = fig.add_subplot(1,3,3, sharey = plot1)
    else:
        plot3 = fig.add_subplot(1,3,3, sharex = plot1)

    
    plot3.imshow(imRGBSeamCarved)

    plt.show()

    #Save seam carved image
    im = Image.fromarray(imRGBSeamCarved)
    im.save('seamcarved.jpg')    

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--testrv':
            imRGBOriginal = imread("Broadway_tower_edit.jpg")
            seamCarve(imRGBOriginal, False, False, 500)
        elif sys.argv[1] == '--testrh':
            imRGBOriginal = imread("kinkakuji.jpg")
            seamCarve(imRGBOriginal, True, False, 200)
        elif sys.argv[1] == '--testev':
            imRGBOriginal = imread("Broadway_tower_edit.jpg")
            seamCarve(imRGBOriginal, False, True, 250)
    else:
        console()