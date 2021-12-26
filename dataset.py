from matplotlib.pyplot import axis
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import torch
import torchaudio
from utils import *
from python_speech_features import delta, mfcc, fbank
import scipy.misc


# class UciDataset(Dataset):
#     def __init__(self):
#         self.data = pd.read_csv(os.path.join("data", "uci", "accent-mfcc-data.csv"))


class KaggleDataset(Dataset):
    def __init__(
        self,
        mode="dev",
        feature="spectrogram",
        classes=KAGGLE_CLASS_NUM,
        segment_length=501,
    ):
        available_modes = ["dev", "train", "test"]
        self.classes = classes
        available_features = [
            "spectrogram",
            "mel_spectrogram",
            "mfcc",
            "fbank",
            "pre_calculated_mfcc",
        ]
        self.segment_length = segment_length
        self.pr = False
        if feature in available_features and mode in available_modes:
            self.data_dict = pd.read_csv(
                os.path.join("data", "kaggle", mode + ".csv"), index_col=0
            )
            self.feature = feature
        else:
            raise

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        file_path = self.data_dict.iloc[idx, 0]
        if not self.feature == "pre_calculated_mfcc":
            waveform, samplerate = torchaudio.load(file_path)
        label = self.data_dict.iloc[idx, 2]
        one_hot_label = torch.zeros(self.classes)
        one_hot_label[label] = 1

        if self.feature == "spectrogram":
            feature = torchaudio.transforms.Spectrogram()(waveform)
        elif self.feature == "mel_spectrogram":
            feature = torchaudio.transforms.MelSpectrogram()(waveform)
        elif self.feature == "mfcc":
            # feature = torchaudio.transforms.MFCC(sample_rate=samplerate)(waveform)
            feature = mfcc(
                waveform, samplerate=samplerate, winlen=0.0025, appendEnergy=False
            )
            delta_mfcc = delta(feature, 1)
            delta_delta_mfcc = delta(delta_mfcc, 1)
            mfccs = np.concatenate((feature, delta_mfcc, delta_delta_mfcc), axis=1)
            feature = np.expand_dims(mfccs.T, 0)
        elif self.feature == "fbank":
            feature = fbank(waveform, samplerate=samplerate, winlen=0.0025)
            feature = np.expand_dims(feature[0].T, 0)
        elif self.feature == "pre_calculated_mfcc":
            feature = np.load(file_path)

        if self.feature == "mfcc":
            z, x, y = feature.shape
        else:
            z, x, y = feature.shape
        if self.pr == False:
            print("Feature Shape", feature.shape)
            self.pr = True
        return (
            feature_reshape(
                torch.tensor(feature_normalization(feature[0]).reshape(1, x, y)),
                self.segment_length,
            ),
            one_hot_label,
        )


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

        return feature_reshape(feature_normalization(feature)), one_hot_label
