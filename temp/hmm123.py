import pandas as pd
import os
import numpy as np
import torchaudio
from python_speech_features import delta, mfcc, fbank
from hmmlearn.hmm import GaussianHMM, MultinomialHMM
features = []
labels = []
data_dict = pd.read_csv(os.path.join("data", "kaggle", "train.csv"), index_col=0)
for i in range(len(data_dict)):
    file_path = data_dict.iloc[i, 0]
    waveform, samplerate = torchaudio.load(file_path.replace(".npy", ".mp3"))
    feature = mfcc(
        waveform, samplerate=samplerate, winlen=0.0025, appendEnergy=False
    )
    delta_mfcc = delta(feature, 1)
    delta_delta_mfcc = delta(delta_mfcc, 1)
    mfccs = np.concatenate((feature, delta_mfcc, delta_delta_mfcc), axis=1)
    label = data_dict.iloc[i, 2]
    labels.append(label)
    features.append(mfccs)

model = MultinomialHMM(n_components=2,  n_iter=1000, verbose=True)
model.fit(np.array(features))
