import os
import numpy as np
import soundfile as sf
from scipy import signal
import librosa
from utils import mix_fixed_SNR


def get_files_and_resample_update(sampling_rate_new, desired_length_seconds, locH, locN, db_SNR=0, mode=0):
    """
    Loads heart sounds and noise, mixes them at fixed SNR, and returns noisy-clean pairs.
    """
    print("========== Starting Data Processing ==========")
    print(f"Mode: {mode}")

    startpath = os.path.abspath(locH)
    noisepath = os.path.abspath(locN)
    class_dirs = os.listdir(startpath)

    print(f"Classes found: {class_dirs}")
    duration_samples = int(desired_length_seconds * sampling_rate_new)

    x_list, y_list, label = [], [], []
    lab = -1
    snr_levels = [-6, -3, 0, 3, 6]

    for class_dir in class_dirs:
        lab += 1
        class_path = os.path.join(startpath, class_dir)
        signal_files = sorted(os.listdir(class_path))
        noise_files = sorted(os.listdir(noisepath))

        for snrx in snr_levels:
            noise_i = 0
            for sig_file in signal_files:
                signal_path = os.path.join(class_path, sig_file)
                noise_path = os.path.join(noisepath, noise_files[noise_i % len(noise_files)])

                signal_data, _ = librosa.load(signal_path, sr=sampling_rate_new)
                noise_data, _ = librosa.load(noise_path, sr=sampling_rate_new,
                                             duration=len(signal_data) / sampling_rate_new)

                k = int(len(signal_data) / duration_samples)
                if len(signal_data) in [9990, 10000, 3500, 4995]:
                    xtem, ytem, ltem = [], [], []
                    for i in range(k):
                        x_segment = signal_data[i * duration_samples: (i + 1) * duration_samples]
                        y_segment = noise_data[i * duration_samples: (i + 1) * duration_samples]

                        real_signal = x_segment / np.max(np.abs(x_segment))
                        noise_signal_raw = y_segment / np.max(np.abs(y_segment))

                        if mode == 0:
                            mixed = mix_fixed_SNR(real_signal, noise_signal_raw, snrx)
                        elif mode == 1:
                            mixed = mix_fixed_SNR(real_signal, noise_signal_raw, db_SNR)
                        else:
                            continue

                        if not np.isnan(mixed.max()):
                            xtem.append(mixed)
                            ytem.append(real_signal)
                            ltem.append(lab)

                    x_list.extend(xtem)
                    y_list.extend(ytem)
                    label.extend(ltem)

                noise_i += 1

    print("========== Finished ==========")
    return np.array(x_list)[..., np.newaxis], np.array(y_list)[..., np.newaxis], np.array(label)