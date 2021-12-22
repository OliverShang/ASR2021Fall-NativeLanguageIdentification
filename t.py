from dataset import CommonVoiceDataset
from torch.utils.data import DataLoader

# from matplotlib.pyplot import pyplot as plt
# Test for dataset

if __name__ == "__main__":
    dev_set = CommonVoiceDataset()
    dev_loader = DataLoader(dev_set, batch_size=64, shuffle=True)
    dev_features, dev_labels = next(iter(dev_loader))
    print(f"Feature batch shape: {dev_features.size()}")
    print(f"Labels batch shape: {dev_labels.size()}")
    print(dev_features[0])
    print(dev_labels[0])
