from simple_cnn_classifier import SimpleARCNN
from dataset import CommonVoiceDataset, KaggleDataset
import torch
from torch.utils.data import DataLoader
from main_aesrc_model import SimpleAESRC
from utils import DEVICE, KAGGLE_CLASS_NUM
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import nni
import tqdm


dev_set = KaggleDataset(feature="mfcc")
dev_loader = DataLoader(dev_set, batch_size=64, shuffle=True, num_workers=4)
test_set = KaggleDataset(feature="mfcc")
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)
asr_loss = torch.nn.CrossEntropyLoss().to(DEVICE)
simpleARCNN = SimpleARCNN(accent_classes=KAGGLE_CLASS_NUM).to(DEVICE)
optimizer = torch.optim.Adam(simpleARCNN.parameters(), lr=0.00001, betas=(0.9, 0.99))


def train():
    for epoch in range(100):#tqdm(range(100)):
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
            y_true = None
            y_pred = None
            for test_data in test_loader:
                mfcc_test, label_test = test_data
                mfcc_test, label_test = mfcc_test.to(DEVICE), label_test.to(DEVICE)
                outputs = simpleARCNN(mfcc_test)
                _, predicted = torch.max(outputs.data, 1)
                total += label_test.size(0)
                acc += (predicted == label_test.argmax(1)).sum()
                y_pred = (
                    predicted.data
                    if y_pred == None
                    else torch.cat((y_pred, predicted.data))
                )
                y_true = (
                    label_test.argmax(1)
                    if y_true == None
                    else torch.cat((y_true, label_test.argmax(1)))
                )
            acc = 100 * acc / total
            print(y_true.shape)
            print(
                confusion_matrix(
                    y_true=y_true.cpu().detach().numpy(),
                    y_pred=y_pred.cpu().detach().numpy(),
                )
            )
            print(acc.item(), "%")


train()
