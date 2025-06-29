import os
import time
import argparse
import tensorflow as tf
import pandas as pd

from Codes.data_loader import get_files_and_resample_update
from Codes.utils import check_SNR_non_merged
from config import *
from model import TBiLSTMModel



def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and measure inference time.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                        help="Device to run inference on (cpu or gpu). Default: cpu")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model weights (.h5 file)")
    parser.add_argument('--output_csv', type=str, default='denoised_recordings.csv',
                        help="Output CSV file for denoised results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    if args.device == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
        device_name = '/CPU:0'
    else:
        device_name = '/GPU:0'

    # Load model
    model = TBiLSTMModel(
        input_shape=input_shape,
        output_shape=output_shape,
        loss_function='mse'
    ).model

    model.load_weights(args.model_path)
    model.summary()

    # Inference loop
    all_estimates = []
    total_time_ms = 0
    total_samples = 0

    for snr in [-6, -3, 0, 3, 6]:
        print(f"\nTesting with SNR = {snr} dB")
        XtestL, YtestL, _ = get_files_and_resample_update(
            1000, 0.8,  # sample rate, split ratio
            locH=pathheartVal,
            locN=pathhospitalval,
            db_SNR=snr,
            mode=1
        )

        with tf.device(device_name):
            start = time.time()
            est_testL = check_SNR_non_merged(XtestL, YtestL, snr, model)
            elapsed_ms = (time.time() - start) * 1000

        all_estimates.extend(est_testL)
        total_time_ms += elapsed_ms
        total_samples += len(XtestL)

        print(f"SNR {snr}: Time = {elapsed_ms:.2f} ms, Samples = {len(XtestL)}, "
              f"Avg per sample = {elapsed_ms / len(XtestL):.2f} ms")

    # Summary
    print("\n=== Inference Summary ===")
    print(f"Device: {args.device.upper()}")
    print(f"Total inference time: {total_time_ms:.2f} ms")
    print(f"Total test samples: {total_samples}")
    print(f"Average time per sample: {total_time_ms / total_samples:.2f} ms")

    # Save results
    df = pd.DataFrame(all_estimates)
    df.to_csv(args.output_csv, index=True)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
