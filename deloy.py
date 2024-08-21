import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.linalg import svd
import scipy as sp 

def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N))
        S = psd(np.fft.rfftfreq(N))
        # Normalize S
        S =  S/np.linalg.norm(S)
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    return f

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))


def add_noise(signal,type_noise =1,alpha =0.01,sample_rate=20000):
    if type_noise == 0:
        return signal
    elif type_noise == 1: #white noise
        signal+= white_noise(len(signal))
    elif type_noise ==2:
        signal+=brownian_noise(len(signal))
    elif type_noise ==3:
        signal+=pink_noise(len(signal))
    else:
        noise,_ =  librosa.load("data/noise1.wav", sr=sample_rate)
        repeats = (len(signal) // len(noise)) + 1  # Số lần lặp lại của source_array cần thiết
        repeated_array = np.tile(noise, repeats)  # Lặp lại source_array nhiều lần
        signal += repeated_array[:len(signal)]
    return signal*alpha


def matrix_to_time_series(matrix, step_size,real_length):
    num_windows, window_size = matrix.shape
    time_series_length = window_size + (num_windows - 1) * step_size
    time_series = np.zeros(time_series_length)
    overlap_count = np.zeros(time_series_length)

    for i in range(num_windows):
        start_index = i * step_size
        end_index = start_index + window_size
        time_series[start_index:end_index] += matrix[i, :]
        overlap_count[start_index:end_index] += 1
    overlap_count[overlap_count == 0] = 1
    
    time_series /= overlap_count
    return time_series[:real_length]
def time_series_to_matrix(time_series, window_size, step_size):
    num_windows = max(1, int(np.ceil((len(time_series) - window_size) / step_size)) + 1)
    matrix = np.zeros((num_windows, window_size))
    
    for i in range(num_windows):
        start_index = i * step_size
        end_index = start_index + window_size
        if end_index <= len(time_series):
            matrix[i, :] = time_series[start_index:end_index]
        else:
            matrix[i, :len(time_series) - start_index] = time_series[start_index:len(time_series)]
            
    return matrix
def generate_k(prercent_k,window_size,step_size):
    return int(min(time_series_to_matrix(np.random.random(1000),window_size,step_size).shape) * prercent_k)
def denoise_signal_svd(x,window_size,step_size,k):
    A = time_series_to_matrix(x, window_size, step_size)
    U, S, Vt = svd(A, full_matrices=False)
    S_p = np.zeros((k, k))
    np.fill_diagonal(S_p, S[:k])
    A_p = U[:, :k] @ S_p @ Vt[:k, :]
    x_p = matrix_to_time_series(A_p,step_size,len(x))
    per = sum(S[:k]) / sum(S)
    return x_p,per

def partitioning_svd(noise_signal,windowsize=256,step=1,k=30):
    t = len(noise_signal)
    i = 0
    p_denoised_signal = np.array([])
    while i < t:
        trim_noise_signal = noise_signal[i:i+1000]
        segment,_ = denoise_signal_svd(trim_noise_signal,windowsize,step,k)
        p_denoised_signal = np.append(p_denoised_signal, segment)
        i+=1000  
    return p_denoised_signal

def filter_fft(signal,sample_rate,low,high):
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), d=1./sample_rate)
    for i,f in enumerate(freq):
        if f < low or f > high :
            fft_spectrum[i] = 0.0
    return np.fft.irfft(fft_spectrum)



    
    













