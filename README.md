# Transformer-BiLSTM Network for Denoising Phonocardiogram Signals

This repository contains the code and other resources deployed in our study:

**"A Robust Deep Learning based Model for Denoising Phonocardiogram Signals in Clinical Environments"**

We propose a novel deep learning architecture for denoising real-world phonocardiogram (PCG) signals. The model, named **T-BiLSTM**, combines a convolutional U-Net backbone with **Bidirectional LSTM layers** and **transformer blocks with multi-head attention** to effectively remove both ambient and physiological noise from heart sound recordings.

To allow e fair benchamrk with the other models published in the literature, some parts of the implementation, i.e., matrices.py, config.py and utils.py , coming from the original LU-Net implementation, were used. This repository was developed by extending and modifying the original **LU-Net** implementation by Shams Nafisa Ali et al.:  
🔗 [https://github.com/ShamsNafisaAli/LU-Net-Heart-Sound-Denoising-](https://github.com/ShamsNafisaAli/LU-Net-Heart-Sound-Denoising-)

We acknowledge and thank the authors for making their code accessible to the community.


## Model Architecture

The following diagram shows the high-level architecture of the proposed **T-BiLSTM** model:
![Model Architecture](figs/T-BiLSTM_model.png)



## Quick Start Guide:

## Dataset Preparation

1. Download the datasets (see `data_link.txt`).
2. Run `preprocessing_data.py` to filter, normalize, and segment the audio.
3. Processed data will be saved in your specified output folder.

## Training

Run `Codes/training_model.py` to start training the T-BiLSTM model on the processed data.

## Testing

Run `Codes/testing_data.py` to evaluate the trained model and compute performance metrics.
