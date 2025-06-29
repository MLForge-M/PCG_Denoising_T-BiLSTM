import librosa
import os
import soundfile as sf
import numpy as np
import scipy.signal as signal


def butter_bandpass_filter(data, lowcut=25, highcut=400, fs=1000, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def remove_spikes(sig, threshold=3.0):
    median = np.median(sig)
    std = np.std(sig)
    spike_indices = np.where(np.abs(sig - median) > threshold * std)[0]
    for i in spike_indices:
        if 1 < i < len(sig) - 2:
            sig[i] = (sig[i-1] + sig[i+1]) / 2.0
    return sig

def spectral_subtraction(noisy_signal, noise_estimate_factor=0.1):
    stft = librosa.stft(noisy_signal)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_mag = np.mean(mag, axis=1, keepdims=True) * noise_estimate_factor
    clean_mag = np.maximum(mag - noise_mag, 0.0)
    cleaned_stft = clean_mag * np.exp(1j * phase)
    return librosa.istft(cleaned_stft)

def normalize_audio(sig):
    return np.clip(sig / np.max(np.abs(sig)), -1.0, 1.0)

def preprocess_signal(signal, sr):
    sig = butter_bandpass_filter(signal, fs=sr)
    sig = remove_spikes(sig)
    sig = spectral_subtraction(sig)
    sig = normalize_audio(sig)
    return sig

def split_audio_into_chunks(input_folder, output_folder, chunk_duration=5):
    os.makedirs(output_folder, exist_ok=True)
    discarded_count = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            print(f"Processing: {filename}")
            path = os.path.join(input_folder, filename)
            audio, sr = librosa.load(path, sr=None)
            audio = preprocess_signal(audio, sr)
            duration = librosa.get_duration(y=audio, sr=sr)

            if duration == 3:
                chunk_filename = f"{os.path.splitext(filename)[0]}_pad.wav"
                sf.write(os.path.join(output_folder, chunk_filename), audio, sr)
            elif duration < 3:
                padding = np.zeros(int(3 * sr) - len(audio))
                padded = np.concatenate((audio, padding))
                chunk_filename = f"{os.path.splitext(filename)[0]}_pad.wav"
                sf.write(os.path.join(output_folder, chunk_filename), padded, sr)
            elif 3 < duration < chunk_duration:
                discarded_count += 1
            else:
                n_chunks = int(duration // chunk_duration)
                for i in range(n_chunks):
                    start = i * chunk_duration * sr
                    end = start + chunk_duration * sr
                    chunk = audio[int(start):int(end)]
                    chunk_filename = f"{os.path.splitext(filename)[0]}_chunk_{i + 1}.wav"
                    sf.write(os.path.join(output_folder, chunk_filename), chunk, sr)

    print("Number of discarded mid-length files:", discarded_count)

if __name__ == "__main__":
    input_folder = "PATH_TO_INPUT_WAVS"
    output_folder = "PATH_TO_OUTPUT_WAVS"
    split_audio_into_chunks(input_folder, output_folder)