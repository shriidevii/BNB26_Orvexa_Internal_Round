import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import soundfile as sf

def compute_fft_spectrum(image_path):
    """Computes the 2D Fast Fourier Transform Magnitude Spectrum."""
    with Image.open(image_path) as img:
        gray = img.convert("L").resize((256, 256))
        arr = np.asarray(gray, dtype=np.float32)

    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor("#231B17")  # Keeps your UI background
    ax.set_facecolor("#231B17")
    
    # Reverted to bright, high-contrast scientific 'inferno'
    ax.imshow(magnitude_spectrum, cmap="inferno") 
    
    ax.set_title("2D-FFT Frequency Spectrum", color="#EDE8E3", fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    return fig

def compute_spectral_energy(audio_path):
    """Computes Short-Time Spectral Energy Distribution."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.patch.set_facecolor("#231B17")  # Keeps your UI background
    ax.set_facecolor("#231B17")

    # Reverted to bright, high-contrast scientific 'viridis'
    ax.specgram(data, Fs=sr, NFFT=512, noverlap=256, cmap="viridis") 
    
    ax.set_title("Acoustic Spectro-Temporal Density", color="#EDE8E3", fontsize=10)
    ax.set_xlabel("Time (s)", color="#CBBFB3", fontsize=8)
    ax.set_ylabel("Frequency (Hz)", color="#CBBFB3", fontsize=8)
    ax.tick_params(colors="#CBBFB3", labelsize=8)
    plt.tight_layout()
    return fig