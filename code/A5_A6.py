import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as image
import scipy.io
import copy

from mpl_toolkits import mplot3d
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, fftn, ifftn

cwd = os.getcwd()
SoundClip = scipy.io.loadmat('CP2_SoundClip.mat')['y'].T[0]
w = int(len(SoundClip.T)/4)
Fs = 44100

L = len(SoundClip)/Fs
n = len(SoundClip)
t = np.arange(0, L , 1/Fs)
k = (1/L)* np.concatenate((np.arange(0, n/2-1+1), np.arange(-n/2,-1+1)))

St = np.fft.fft(SoundClip)
bass_thresholds = 250

bassindex = []
abs_k = np.abs(k)
for i in range(len(k)):
    if abs_k[i]<=250:
        bassindex = np.append(bassindex,int(i)).astype(np.int64)
    else:
        continue
        
bassline_t = np.zeros(np.shape(St),dtype=np.complex_)
for i in bassindex:
    bassline_t[i]=St[i]
    
bassline = np.fft.fftshift(np.abs(np.fft.ifft(bassline_t)))
A5 = copy.copy(bassline.reshape((len(bassline),1)))

guitarindex = []
abs_k = np.abs(k)
for i in range(len(k)):
    if abs_k[i]<=1200 and abs_k[i]>=80:
        guitarindex = np.append(guitarindex,int(i)).astype(np.int64)
    else:
        continue
        
guitar_t = np.zeros(np.shape(St),dtype=np.complex_)
for i in guitarindex:
    guitar_t[i]=St[i]
    
guitar = np.fft.fftshift(np.abs(np.fft.ifft(guitar_t)))
A6 = copy.copy(guitar.reshape((len(guitar),1)))

banjoindex = []
abs_k = np.abs(k)
for i in range(len(k)):
    if abs_k[i]<=588 and abs_k[i]>=294:
        banjoindex = np.append(banjoindex,int(i)).astype(np.int64)
    else:
        continue
        
banjo_t = np.zeros(np.shape(St),dtype=np.complex_)
for i in banjoindex:
    banjo_t[i]=St[i]
    
banjo = np.fft.fftshift(np.abs(np.fft.ifft(banjo_t)))




