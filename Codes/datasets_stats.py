import glob
import librosa
import matplotlib.pyplot as plt
import os
import pandas as pd


def describe_signal_lengths(data):
    """
    This function takes a dictionary of signal lengths and prints the name of each key
    along with the mean, maximum, and minimum values for each key.
    """
    for key, values in data.items():
        mean_value = sum(values) / len(values)
        max_value = max(values)
        min_value = min(values)

        print(f"Class: {key}")
        print(f"Mean: {mean_value}")
        print(f"Max: {max_value}")
        print(f"Min: {min_value}")
        print("")


#hsct11
def dataset_hsct11(file_path):
    df = {'hsct': []}
    sampling_freqs = []
    catName = 'hsct'
    for fn in glob.glob(os.path.join(file_path, '*.wav')):
        input_file, sr = librosa.load(fn, sr=None)
        length = librosa.get_duration(y=input_file, sr=sr)
        length = round(length, 2)
        df[catName].append(length)
        if not sampling_freqs:
            sampling_freqs.append(sr)
        else:
            if sr not in sampling_freqs:
                sampling_freqs.append(sr)
    return df, sampling_freqs


#PHHS
def dataset_phhs(file_path):
    df = {'PHHS': []}
    sampling_freqs = []
    catName = 'PHHS'
    for fn in glob.glob(os.path.join(file_path, '*.wav')):
        #print(fn)
        input_file, sr = librosa.load(fn, sr=None)
        length = librosa.get_duration(y=input_file, sr=sr)
        length = round(length, 2)
        df[catName].append(length)
        if not sampling_freqs:
            sampling_freqs.append(sr)
        else:
            if sr not in sampling_freqs:
                sampling_freqs.append(sr)
    return df, sampling_freqs


#physionet-2016-training
def dataset_physionet2016(file_path):
    df = {'training-a': [], 'training-b': [], 'training-c': [], 'training-d': [], 'training-e': [], 'training-f': []}
    for root, dirs, files in os.walk(file_path):
        if not dirs:
            catName = root.split('\\')[-1]
            for filename in files:
                if filename.endswith(".wav"):
                    #print(filename)
                    input_file, sr = librosa.load(os.path.join(root,filename), sr=None)
                    length = librosa.get_duration(y=input_file, sr=sr)
                    df[catName].append(length)
    return df


#Physionet-22
def dataset_physionet22(file_path):
    df = {'physionet': []}
    sampling_freqs = []
    catName = 'physionet'
    for fn in glob.glob(os.path.join(file_path, '*.wav')):
        #print(fn)
        input_file, sr = librosa.load(fn, sr=None)
        length = librosa.get_duration(y=input_file, sr=sr)
        length = round(length, 2)
        df[catName].append(length)
        if not sampling_freqs:
            sampling_freqs.append(sr)
        else:
            if sr not in sampling_freqs:
                sampling_freqs.append(sr)
    return df, sampling_freqs


if __name__ == "__main__":

    """
    Description of hsct11
    """
    df_hsct11, sf_hsct11 = dataset_hsct11(r'train/hsct11')

    pd_df = pd.DataFrame(df_hsct11)
    pd_df.hist()  # bins=5
    plt.title('Histogram of Occurrences of Signal Length')
    plt.xlabel('Signal Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    describe_signal_lengths(df_hsct11)

    """
    Description of PHHS
    """
    df_phhs, sf_phhs = dataset_phhs(r'train/PHHS')

    pd_df = pd.DataFrame(df_phhs)
    pd_df.hist()
    plt.title('Histogram of Occurrences of Signal Length')
    plt.xlabel('Signal Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    describe_signal_lengths(df_phhs)

    """
    Description of physionet-2016-training
    """
    df_physio2016 = dataset_physionet2016(r'train/physionet-2016-training')

    pd_df = pd.DataFrame(df_physio2016['training-a'])  #'training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f'
    pd_df.hist()
    plt.title('Histogram of Occurrences of Signal Length')
    plt.xlabel('Signal Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    describe_signal_lengths(df_physio2016)

    """
    Description of Physionet-22
    """
    df_physio22, sf_physio22 = dataset_physionet22(r'train/physionet22')

    pd_df = pd.DataFrame(df_physio22)
    pd_df.hist()
    plt.title('Histogram of Occurrences of Signal Length')
    plt.xlabel('Signal Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    describe_signal_lengths(df_physio22)