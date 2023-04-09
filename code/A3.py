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

S1 = SoundClip[(1-1)*w : 1*w]
S2 = SoundClip[(2-1)*w : 2*w]
S3 = SoundClip[(3-1)*w : 3*w]
S4 = SoundClip[(4-1)*w : 4*w]

L = len(S1)/ Fs
n = len(S1)
t = np.arange(0, L , 1/Fs)
tau = np.arange(0, L, 0.1)
k = 2*np.pi *(1/L/2)* np.concatenate((np.arange(0, n/2-1+1), np.arange(-n/2,-1+1)))
ks = np.fft.fftshift(k)
Sgt_spec = np.zeros((len(ks),len(tau)))


##S3 spectrogram

a = 400
S3gt_spec = np.zeros((len(ks),len(tau)))
for i in range(len(tau)):
    g = np.exp(-a*(t-tau[i])**2)
    S3g = g*S3
    S3gt = np.fft.fft(S3g)
    
    loc_i = np.argmax(np.abs(S3gt)[0:1800])
    max_k = np.abs(k[loc_i])
    
    filter_g = np.exp(-(1/L)*(np.abs(k)-max_k)**2)
    filter_S3g = filter_g*S3gt
    S3gt_spec[:,i] = np.fft.fftshift(np.abs(filter_S3g))
    
A3 = copy.copy(S3gt_spec)