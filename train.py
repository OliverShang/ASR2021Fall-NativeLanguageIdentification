from simple_cnn_classifier import SimpleARCNN
from dataset import CommonVoiceDataset
import torch
from torch.utils.data import DataLoader

from utils import DEVICE



dev_set = CommonVoiceDataset(feature="spectrogram")
dev_loader = DataLoader(dev_set, batch_size=64, shuffle=True, num_workers=4)
test_set = CommonVoiceDataset(feature="spectrogram")
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)
asr_loss = torch.nn.MSELoss().to(DEVICE)
simpleARCNN = SimpleARCNN(7).to(DEVICE)
optimizer = torch.optim.Adam(simpleARCNN.parameters(), lr=0.00001, betas=(0.9, 0.99))

def train():
    for epoch in range(10):
        for i, (mfcc, label) in enumerate(dev_loader):
            mfcc_gpu = mfcc.to(DEVICE)
            label_gpu = label.to(DEVICE)
            optimizer.zero_grad()
            loss = asr_loss(simpleARCNN(mfcc_gpu), label_gpu)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            acc = 0
            total = 0
            for test_data in test_loader:
                mfcc_test, label_test = test_data
                mfcc_test, label_test = mfcc_test.to(DEVICE), label_test.to(DEVICE)
                outputs = simpleARCNN(mfcc_test)
                _, predicted = torch.max(outputs.data, 1)
                total += label_test.size(0)
                acc += (predicted == label_test.argmax(1)).sum()
            acc = 100 * acc / total
            print(acc.item(), "%")
        
train()