import numpy as np

def Frame(Y,framesize):
    totalsamples = np.shape(Y)[0]
    numframes = int(np.ceil(totalsamples/framesize))

    outputY = np.zeros((numframes,framesize,np.shape(Y)[1]))
    for frame_i in range(numframes-1):
        tmp = Y[frame_i*framesize:(frame_i+1)*framesize,:]
        outputY[frame_i,:,:] = tmp

    remain = totalsamples%framesize
    if remain==0:
        remain = framesize

    outputY[-1,0:remain,:] = Y[(numframes-1)*framesize:,:]

    return outputY