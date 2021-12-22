from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torchaudio


class CommonVoiceDataset(Dataset):
    def __init__(self, mode="dev", feature="spectrogram"):
        available_modes = ["dev", "train", "test"]
        available_features = ["spectrogram", "mel_spectrogram", "mfcc"]
        if mode in available_modes and feature in available_features:
            # print(111)
            self.data_dict = pd.read_csv(
                os.path.join("data", mode + "_filtered.tsv"), sep="\t", usecols=[2, 11]
            )
            self.feature = feature
        else:
            raise

    def __len__(self):
        return len(self.data_dict)

    def feature_reshape(feature, max_len=1200):
        h, w = feature.shape
        if h >= max_len:
            return feature[:max_len]
        else:
            feat_ = np.zeros((max_len, w))
            feat_[:h] = feature
            return feat_

    def __getitem__(self, idx):
        file_path = os.path.join("data", "clips", self.data_dict.iloc[idx, 0])
        waveform, sample_rate = torchaudio.load(file_path)
        feature = waveform
        label = self.data_dict.iloc[idx, 1]
        if self.feature == "spectrogram":
            feature = torchaudio.transforms.Spectrogram()(waveform)
        elif self.feature == "mel_spectrogram":
            feature = torchaudio.transforms.MelSpectrogram()(waveform)
        elif self.feature == "mfcc":
            feature = torchaudio.transforms.MFCC()(waveform)

        return self.feature_reshape(feature), label
