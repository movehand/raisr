import cv2
import numpy as np
import os
import pickle
import sys
from scipy.misc import imresize
from cgls import cgls
from filterplot import filterplot
from gaussian2d import gaussian2d
from gettrainargs import gettrainargs
from hashkey import hashkey
from math import floor
from matplotlib import pyplot as plt
from scipy import interpolate

args = gettrainargs()

# Define parameters
R = 2
patchsize = 11
gradientsize = 9
Qangle = 24
Qstrength = 3
Qcoherence = 3
trainpath = 'train'

# Calculate the margin
maxblocksize = max(patchsize, gradientsize)
margin = floor(maxblocksize/2)
patchmargin = floor(patchsize/2)
gradientmargin = floor(gradientsize/2)

Q = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize, patchsize*patchsize))
V = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))
h = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))
mark = np.zeros((Qstrength*Qcoherence, Qangle, R*R))

# Matrix preprocessing
# Preprocessing normalized Gaussian matrix W for hashkey calculation
weighting = gaussian2d([gradientsize, gradientsize], 2)
weighting = np.diag(weighting.ravel())

# Get image list
imagelist = []
for parent, dirnames, filenames in os.walk(trainpath):
    for filename in filenames:
        if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            imagelist.append(os.path.join(parent, filename))

# Compute Q and V
imagecount = 1
for image in imagelist:
    print('\r', end='')
    print(' ' * 60, end='')
    print('\rProcessing image ' + str(imagecount) + ' of ' + str(len(imagelist)) + ' (' + image + ')')
    origin = cv2.imread(image)
    # Extract only the luminance in YCbCr
    grayorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)[:,:,0]
    # Normalized to [0,1]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
    # Downscale (bicubic interpolation)
    height, width = grayorigin.shape
    LR = imresize(grayorigin, (floor((height+1)/2),floor((width+1)/2)), interp='bicubic', mode='F')
    # Upscale (bilinear interpolation)
    height, width = LR.shape
    heightgrid = np.linspace(0, height-1, height)
    widthgrid = np.linspace(0, width-1, width)
    bilinearinterp = interpolate.interp2d(widthgrid, heightgrid, LR, kind='linear')
    heightgrid = np.linspace(0, height-1, height*2-1)
    widthgrid = np.linspace(0, width-1, width*2-1)
    upscaledLR = bilinearinterp(widthgrid, heightgrid)
    # Calculate A'A, A'b and push them into Q, V
    height, width = upscaledLR.shape
    operationcount = 0
    totaloperations = (height-2*margin) * (width-2*margin)
    for row in range(margin, height-margin):
        for col in range(margin, width-margin):
            if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                print('\r|', end='')
                print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                sys.stdout.flush()
            operationcount += 1
            # Get patch
            patch = upscaledLR[row-patchmargin:row+patchmargin+1, col-patchmargin:col+patchmargin+1]
            patch = np.matrix(patch.ravel())
            # Get gradient block
            gradientblock = upscaledLR[row-gradientmargin:row+gradientmargin+1, col-gradientmargin:col+gradientmargin+1]
            # Calculate hashkey
            angle, strength, coherence = hashkey(gradientblock, Qangle, weighting)
            # Get pixel type
            pixeltype = ((row-margin) % R) * R + ((col-margin) % R)
            # Get corresponding HR pixel
            pixelHR = grayorigin[row,col]
            # Compute A'A and A'b
            ATA = np.dot(patch.T, patch)
            ATb = np.dot(patch.T, pixelHR)
            ATb = np.array(ATb).ravel()
            # Compute Q and V
            Q[angle,strength,coherence,pixeltype] += ATA
            V[angle,strength,coherence,pixeltype] += ATb
            mark[coherence*3+strength, angle, pixeltype] += 1
    imagecount += 1

# Preprocessing permutation matrices P for nearly-free 8x more learning examples
print('\r', end='')
print(' ' * 60, end='')
print('\rPreprocessing permutation matrices P for nearly-free 8x more learning examples ...')
sys.stdout.flush()
P = np.zeros((patchsize*patchsize, patchsize*patchsize, 7))
rotate = np.zeros((patchsize*patchsize, patchsize*patchsize))
flip = np.zeros((patchsize*patchsize, patchsize*patchsize))
for i in range(0, patchsize*patchsize):
    i1 = i % patchsize
    i2 = floor(i / patchsize)
    j = patchsize * patchsize - patchsize + i2 - patchsize * i1
    rotate[j,i] = 1
    k = patchsize * (i2 + 1) - i1 - 1
    flip[k,i] = 1
for i in range(1, 8):
    i1 = i % 4
    i2 = floor(i / 4)
    P[:,:,i-1] = np.linalg.matrix_power(flip,i2).dot(np.linalg.matrix_power(rotate,i1))
Qextended = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize, patchsize*patchsize))
Vextended = np.zeros((Qangle, Qstrength, Qcoherence, R*R, patchsize*patchsize))
for pixeltype in range(0, R*R):
    for angle in range(0, Qangle):
        for strength in range(0, Qstrength):
            for coherence in range(0, Qcoherence):
                for m in range(1, 8):
                    m1 = m % 4
                    m2 = floor(m / 4)
                    newangleslot = angle
                    if m2 == 1:
                        newangleslot = Qangle-angle-1
                    newangleslot = int(newangleslot-Qangle/2*m1)
                    while newangleslot < 0:
                        newangleslot += Qangle
                    newQ = P[:,:,m-1].T.dot(Q[angle,strength,coherence,pixeltype]).dot(P[:,:,m-1])
                    newV = P[:,:,m-1].T.dot(V[angle,strength,coherence,pixeltype])
                    Qextended[newangleslot,strength,coherence,pixeltype] += newQ
                    Vextended[newangleslot,strength,coherence,pixeltype] += newV
Q += Qextended
V += Vextended

# Compute filter h
print('Computing h ...')
sys.stdout.flush()
operationcount = 0
totaloperations = R * R * Qangle * Qstrength * Qcoherence
for pixeltype in range(0, R*R):
    for angle in range(0, Qangle):
        for strength in range(0, Qstrength):
            for coherence in range(0, Qcoherence):
                if round(operationcount*100/totaloperations) != round((operationcount+1)*100/totaloperations):
                    print('\r|', end='')
                    print('#' * round((operationcount+1)*100/totaloperations/2), end='')
                    print(' ' * (50 - round((operationcount+1)*100/totaloperations/2)), end='')
                    print('|  ' + str(round((operationcount+1)*100/totaloperations)) + '%', end='')
                    sys.stdout.flush()
                operationcount += 1
                h[angle,strength,coherence,pixeltype] = cgls(Q[angle,strength,coherence,pixeltype], V[angle,strength,coherence,pixeltype])

# Write filter to file
with open("filter", "wb") as fp:
    pickle.dump(h, fp)

# Plot the learned filters
if args.plot:
    filterplot(h, R, Qangle, Qstrength, Qcoherence, patchsize)

print('\r', end='')
print(' ' * 60, end='')
print('\rFinished.')
