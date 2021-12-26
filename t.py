from dataset import CommonVoiceDataset, KaggleDataset
from torch.utils.data import DataLoader
import torchaudio

# from matplotlib.pyplot import pyplot as plt
# Test for dataset

if __name__ == "__main__":
    # dev_set = CommonVoiceDataset(feature="mfcc")
    # dev_loader = DataLoader(dev_set, batch_size=64, shuffle=True)
    # dev_features, dev_labels = next(iter(dev_loader))

    dev_set = KaggleDataset(feature="fbank")
    dev_loader = DataLoader(dev_set, batch_size=64, shuffle=True)
    dev_features, dev_labels = next(iter(dev_loader))
    print(f"Feature batch shape: {dev_features.size()}")
    print(f"Labels batch shape: {dev_labels.size()}")
    print(dev_features[0])
    print(dev_labels[0])

    # waveform, samplerate = torchaudio.load("data/kaggle/korean/korean14.mp3")
    # feature = torchaudio.transforms.MFCC(sample_rate=samplerate, n_mels=64)(waveform)
    # print(feature.shape)
