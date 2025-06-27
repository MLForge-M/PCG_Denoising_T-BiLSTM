# T-BiLSTM: Transformer-Enhanced BiLSTM Model for Phonocardiogram Signal Denoising

This repository contains the implementation of **T-BiLSTM**, a deep learning model developed for the denoising of phonocardiogram (PCG) signals in real-world clinical conditions. Our method extends the LU-Net architecture proposed in [Ali et al., 2023](https://github.com/ShamsNafisaAli/LU-Net-Heart-Sound-Denoising-) 

The model is designed to suppress both ambient and internal noise while preserving key cardiac signal components such as S1 and S2 sounds. It is trained and evaluated on five diverse PCG datasets, ensuring generalization across clinical scenarios.

## Key Features

- Hybrid U-Net architecture enhanced with BiLSTM and Transformer layers.
- Optimized for denoising heart sound signals at low SNR levels (e.g., 6 dB).
- Evaluated on extended datasets (e.g., OAHS+ICBHI, OAHS-HAN).
- Suitable for near real-time applications in resource-constrained environments.
- Outperforms baseline LU-Net and U-Net models in denoising metrics (SNR, PRD, RMSE).

## Model Architecture

The following diagram shows the high-level architecture of the proposed **T-BiLSTM** model:
![Model Architecture](figs/T-BiLSTM_model.png)

Each decoder block after Decoder₅ receives the concatenated output features from the previous block and the corresponding transformer block. This enables context-aware refinement of the reconstructed signal. Skip connections are used to preserve spatial information and help model convergence. The final decoding layer uses a 1D convolution with `tanh` activation to reconstruct the clean heart sound waveform.

## Repository Structure

