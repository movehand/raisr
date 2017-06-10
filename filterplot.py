import numpy as np
from matplotlib import pyplot as plt

def filterplot(h, R, Qangle, Qstrength, Qcoherence, patchsize):
    for pixeltype in range(0,R*R):
        maxvalue = h[:,:,:,pixeltype].max()
        minvalue = h[:,:,:,pixeltype].min()
        fig = plt.figure(pixeltype)
        plotcounter = 1
        for coherence in range(0, Qcoherence):
            for strength in range(0, Qstrength):
                for angle in range(0, Qangle):
                    filter1d = h[angle,strength,coherence,pixeltype]
                    filter2d = np.reshape(filter1d, (patchsize, patchsize))
                    ax = fig.add_subplot(Qstrength*Qcoherence, Qangle, plotcounter)
                    ax.imshow(filter2d, interpolation='none', extent=[0,10,0,10], vmin=minvalue, vmax=maxvalue)
                    ax.axis('off')
                    plotcounter += 1
        plt.axis('off')
        plt.show()
