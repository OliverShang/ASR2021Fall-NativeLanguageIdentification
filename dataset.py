from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from utils import *
from python_speech_features import delta


# class UciDataset(Dataset):
#     def __init__(self):
#         self.data = pd.read_csv(os.path.join("data", "uci", "accent-mfcc-data.csv"))


def feature_reshape(feature, max_len=1200):
    x, y, z = feature.shape
    if z >= max_len:
        return feature[:, :, :max_len]
    else:
        feat_ = torch.zeros((x, y, max_len))
        feat_[:, :, :z] = feature
        return feat_


class KaggleDataset(Dataset):
    def __init__(self, feature="spectrogram"):
        available_features = ["spectrogram", "mel_spectrogram", "mfcc"]
        if feature in available_features:
            self.data_dict = pd.read_csv(
                os.path.join("data", "kaggle", "dev.csv"), index_col=0
            )
            self.feature = feature
        else:
            raise

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        file_path = self.data_dict.iloc[idx, 0]
        waveform, samplerate = torchaudio.load(file_path)
        label = self.data_dict.iloc[idx, 2]
        one_hot_label = torch.zeros(KAGGLE_CLASS_NUM)
        one_hot_label[label] = 1

        if self.feature == "spectrogram":
            feature = torchaudio.transforms.Spectrogram(sample_rate=samplerate)(
                waveform
            )
        elif self.feature == "mel_spectrogram":
            feature = torchaudio.transforms.MelSpectrogram(sample_rate=samplerate)(
                waveform
            )
        elif self.feature == "mfcc":
            feature = torchaudio.transforms.MFCC(sample_rate=samplerate)(waveform)

        return feature_reshape(feature), one_hot_label


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

    def feature_reshape(self, feature, max_len=1200):
        x, y, z = feature.shape
        if z >= max_len:
            return feature[:, :, :max_len]
        else:
            feat_ = torch.zeros((x, y, max_len))
            feat_[:, :, :z] = feature
            return feat_

    def __getitem__(self, idx):
        file_path = os.path.join("data", "clips", self.data_dict.iloc[idx, 0])
        waveform, sample_rate = torchaudio.load(file_path)
        feature = waveform
        label = self.data_dict.iloc[idx, 1]
        one_hot_label = torch.zeros(CLASS_NUM)
        one_hot_label[label] = 1
        if self.feature == "spectrogram":
            feature = torchaudio.transforms.Spectrogram()(waveform)
        elif self.feature == "mel_spectrogram":
            feature = torchaudio.transforms.MelSpectrogram()(waveform)
        elif self.feature == "mfcc":
            feature = torchaudio.transforms.MFCC()(waveform)
            # print(feature.shape)

        return feature_reshape(feature), one_hot_label
