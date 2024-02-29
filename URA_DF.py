from numpy import cos, sin, pi, absolute, arange
import numpy as np
from scipy.signal import kaiserord, lfilter, firwin, freqz, upfirdn, group_delay, correlate
from scipy.optimize import minimize
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
import frame, URAfun
from scipy.io import wavfile
import matplotlib.pyplot as plt
import cv2

wav_fname = 'record_comp090.wav'
video_name = 'record_comp090.avi'
samplerate, Y = wavfile.read(wav_fname)

p1 = np.transpose(np.array([[-21,-63,0]]))*1e-3
p2 = np.transpose(np.array([[-63,-63,0]]))*1e-3
p3 = np.transpose(np.array([[-21,-21,0]]))*1e-3
p4 = np.transpose(np.array([[-63,-21,0]]))*1e-3
p5 = np.transpose(np.array([[-21,21,0]]))*1e-3
p6 = np.transpose(np.array([[-63,21,0]]))*1e-3
p7 = np.transpose(np.array([[-21,63,0]]))*1e-3
p8 = np.transpose(np.array([[-63,63,0]]))*1e-3
p9 = np.transpose(np.array([[63,63,0]]))*1e-3
p10 = np.transpose(np.array([[21,63,0]]))*1e-3
p11 = np.transpose(np.array([[63,21,0]]))*1e-3
p12 = np.transpose(np.array([[21,21,0]]))*1e-3
p13 = np.transpose(np.array([[63,-21,0]]))*1e-3
p14 = np.transpose(np.array([[21,-21,0]]))*1e-3
p15 = np.transpose(np.array([[63,-63,0]]))*1e-3
p16 = np.transpose(np.array([[21,-63,0]]))*1e-3
mic_pos = np.concatenate([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16],axis=1)

# The Nyquist rate of the signal.
speed = 340
nyq_rate = samplerate / 2.0
drone_freq_range = [1500, 4500]
normalized_drone_freq_range = [drone_freq_range[0]/nyq_rate, drone_freq_range[1]/nyq_rate]
# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.  We'll design the filter
# with a 10 Hz transition width.
width = 0.11
# The desired attenuation in the stop band, in dB.
ripple_db = 60.0
# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)
# Use firwin with a Kaiser window to create a FIR filter.
bandtaps = firwin(N, normalized_drone_freq_range, window=('kaiser', beta), pass_zero=False)
hightaps = firwin(N, normalized_drone_freq_range[0], window=('kaiser', beta), pass_zero=False)

# Use lfilter to filter x with the FIR filter.
for i in range(np.shape(Y)[1]):
    Y[:,i] = lfilter(hightaps, 1.0, Y[:,i])

bufferLength = 1024
# numBuffers x bufferLength x Channels
framedY = frame.Frame(Y,bufferLength)*1000


interpFactor = 8
# Interpolator usually uses a low-pass filter as numberator
interpb = interpFactor*firwin((2*interpFactor*8-1), 1/interpFactor, pass_zero=True)
# upfirdn(interpb, x,up=interpFactor)
# FIRInterpolator introduces group delay into the signal.
_, gd = group_delay((interpb, 1))
groupDelay = np.median(gd)
#print(groupDelay)

azimuth_list = []
elevation_list = []

for fi in range(np.shape(framedY)[0]-1):
    timematrix = np.zeros((16,16))
    for i in range(16):
        for j in range(16):
            if i!=j:
                xcCoarse = correlate(framedY[fi,:,i],framedY[fi,:,j])
                xcDense = upfirdn(interpb, xcCoarse,up=interpFactor)

                idxloc = np.argmax(xcDense); #positive index means Y(:,j) is ahead of Y(:,i) with time idx
                timematrix[i,j] = (idxloc - groupDelay)/interpFactor - bufferLength
    
    timematrix = timematrix-np.mean(timematrix,1)
    timevector = np.mean(timematrix,0,keepdims=True)*(1/samplerate)
    distvector = timevector*speed

    lb = np.deg2rad((-180,0))
    ub = np.deg2rad((180,60))
    bnds = np.array([[lb[0], ub[0]],[lb[1], ub[1]]])
    cons = ()
    res = minimize(lambda x:URAfun.efun(x,mic_pos,distvector), ((lb[0]+ub[0])/2, (lb[1]+ub[1])/2), bounds=bnds,constraints=cons)
    azimuth_list.append(res.x[0])
    elevation_list.append(res.x[1])

frames = []
fig = plt.figure(1)
ax = fig.add_axes([0.1,0.1,0.8,0.8], polar=True)
ax.set_rlim(90, 0)
for a,e in zip(azimuth_list,elevation_list):
    ax.plot(a, np.rad2deg(e), 'ro')
    canvas = fig.canvas
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)
    frames.append(image)
    ax.clear()
    ax.set_rlim(90, 0)

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MJPG'), int(samplerate/bufferLength), (np.shape(frames[0])[1],np.shape(frames[0])[0]))
for frame in frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video.write(frame)
video.release()

figure(2)
clf()
w, h = freqz(interpb, worN=8000)
plot((w/pi)*nyq_rate, 20*np.log10(absolute(h)), linewidth=2)
xlabel('Frequency (Hz)')
ylabel('Gain (db)')
title('Frequency Response')
grid(True)

show()