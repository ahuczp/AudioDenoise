from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import soundfile as sf
import wave

if __name__=='__main__':
    filename = "C:/Users/czp/Desktop/audio_deniose/Audio-Denoising-master/sample.wav"
    f = wave.open(filename, "rb")

    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    Fe = framerate
    str_data = f.readframes(nframes)
    f.close()
    x = np.fromstring(str_data,dtype = np.int16)
    x = x[100000-1:]
    Nx = len(x)
    print(Nx)

    #parameters
    apriori_SNR = 1 # select 0 for aposteriori SNR estimation and 1 for apriori
    alpha = 0.05  # only used if apriori_SNR=1
    beta1 = 0.5
    beta2 = 1
    lamb = 3

    # STFT parameters
    NFFT = 1024
    window_length = round(0.031 * Fe)
    window = np.hamming(window_length)
    #window = window[:]
    overlap = np.floor(0.45 * window_length)  # number of windows samples without overlapping

    # Signal parameters
    #t_min = 0.4  # interva for learning the noise
    #t_max = 1.00  # spectrum( in second)
    print(type(x))
    x = [item+2.2204e-16j for item in x]
    for i in range(10):
        print(x[i])
    x  =np.array(x)
    F,T,S = signal.spectrogram(x, fs=Fe, window = window, noverlap=window_length-overlap,nfft=NFFT)
    Nf,Nw = S.shape    #(343, 1576)
    print(S.shape)     #matlab (1024 ,1935)

    t_start = -1
    t_end = -1
    for i in range(len(T)):
        if(T[i]>0.40):
            t_start = i
            break
    for i in range(len(T)):
        if(T[i]>1.0):
            t_end = i
            break

    #print('t_start: ',t_start)
    #print('t_end: ',t_end)
    abS_noise = abs(S[:,t_start:t_end])**2
    print(abS_noise[0][0])
    print("abS_noise: ",abS_noise.shape)
    print(type(abS_noise[0][0]))
    #print(type(abS_noise))
    noise_spectrum = np.mean(abS_noise,1)
    print('noise_spectrum : ',noise_spectrum.shape)


    #noise_spectram = np.tile(noise_spectrum.T,(1,int(Nw/abS_noise.shape[1])+1))
    noise_spectram = np.tile(noise_spectrum, (Nw, 1))  #这个地方可能存在问题，可能需要反转一次
    noise_spectram = noise_spectram.T
    print(noise_spectram.shape)

    absS = abs(S)**2
    print(absS.shape)

    #SNR_est =np.zeros(absS.shape)
    #an_lk = np.zeros(absS.shape)
    STFT = np.zeros(absS.shape)
    for i in range(absS.shape[0]):
        for j in range(absS.shape[1]):
            SNR_est = max((float(absS[i][j])/ noise_spectram[i][j]) - 1, 0)
            an_lk = max((1-lamb*((1.0/(SNR_est+1))**beta1))**beta2,0)
            STFT[i][j] = an_lk*S[i][j]
    #an_lk = max((1 -lambda * ((1. / (SNR_est+1)). ^ beta1)).^ beta2, 0);
    C:\Users\czp\Desktop\czp\ml\audio_denoise\venv\denoise1.py
    #print(an_lk.shape)
    #ind = mod((1:window_length) - 1, Nf) + 1;
    ind1 = range(window_length)
    ind = [(item%Nf)+1 for item in ind1]
    output_signal = np.zeros(int((Nw -1) * overlap + window_length))
    print(output_signal.shape)

    '''s
    % for indice=1:Nw % Overlap addtechnique
    % left_index = ((indice - 1) * overlap);
    % index = left_index + [1:window_length];
    % temp_ifft = real(ifft(STFT(:, indice), NFFT));
    % output_signal(index) = output_signal(index) + temp_ifft(ind). * window;
    % end
    '''

    '''
    for indice in range(Nw):
        left_index = (indice*overlap)
        index = [left_index+item for item in ind1]
        temp_ifft = real(ifft(STFT[:, index[0]:index[-1]], NFFT))
        output_signal[index[0]:index[-1]] = output_signal(index) + temp_ifft(ind). * window
    '''