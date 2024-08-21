import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.linalg import svd
import scipy as sp 
from scipy.io.wavfile import write
from deloy import generate_k, partitioning_svd, filter_fft, denoise_signal_svd
import plotly.graph_objects as go

st.title("Denoise using SVD project")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    samples, sample_rate = librosa.load(uploaded_file, sr=None)

    fig = go.Figure()

    fig.add_trace(go.Scatter( y=samples, mode='lines', name='Original'))

    fig.update_layout(
        title='Original audio',
        xaxis_title='Time',
        yaxis_title='Amplitude',
        xaxis_rangeslider_visible=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.audio(uploaded_file, format="audio/wav")

    window_size = st.number_input("Window size:", value=256, placeholder="Type a window size...")
    step_size = st.number_input("Step size:", value=1, placeholder="Type a step size...")
    percent_k = st.slider("Choose percent_k:", 0.0, 1.0, 0.5)
    k = generate_k(percent_k, window_size, step_size)
    low_filter_threshold = st.slider("Choose low-threshold of filter:", 1, 100, 0, step=10)
    high_filter_threshold = st.slider("Choose high-threshold of filter:", 20000, 40000, 1000, step=100)

    if st.button("Denoise"):
        denoised_audio = partitioning_svd(samples, window_size, step_size, k)
        denoised_audio = filter_fft(denoised_audio, sample_rate, low_filter_threshold, high_filter_threshold)

        fig = go.Figure()

        fig.add_trace(go.Scatter(y=samples, mode='lines', name='Original'))
        fig.add_trace(go.Scatter(y=denoised_audio, mode='lines', name='Denoised'))

        fig.update_layout(
            title='Denoised audio',
            xaxis_title='Time',
            yaxis_title='Amplitude',
            xaxis_rangeslider_visible=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.audio(denoised_audio, sample_rate=sample_rate)