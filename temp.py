from torch.optim import optimizer
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

batch_size = 8
train_set = KaggleDataset(mode="train", feature="spectrogram")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = KaggleDataset(mode="test", feature="spectrogram")
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
asr_loss = torch.nn.CrossEntropyLoss().to(DEVICE)
simpleARCNN = SimpleAESRC(2).to(DEVICE)
optimizer = torch.optim.Adam(simpleARCNN.parameters(), lr=0.0001)

# optimizer = torch.optim.SGD(simpleARCNN.parameters(), lr=0.001)


def train():
    for epoch in range(100):
        with tqdm.tqdm(total=int(len(train_set) / batch_size), desc="Training") as t:
            total_loss = 0
            total_iter = 0
            total_acc = 0
            for i, (mfcc, label) in enumerate(train_loader):
                mfcc_gpu = mfcc.to(DEVICE)
                label_gpu = label.to(DEVICE)
                optimizer.zero_grad()
                outputs = simpleARCNN(mfcc_gpu)
                loss = asr_loss(outputs, label_gpu)
                total_loss += loss.cpu().detach().numpy()
                _, predicted = torch.max(outputs.data, 1)
                total_acc += (predicted == label_gpu).sum()
                loss.backward()
                optimizer.step()
                total_iter = total_iter + batch_size
                t.set_postfix(
                    loss=total_loss / total_iter,
                    acc=total_acc.cpu().numpy() / total_iter,
                )
                t.update(1)
        with torch.no_grad():
            acc = 0
            total = 0
            y_true = None
            y_pred = None
            for test_data in test_loader:
                mfcc_test, label_test = test_data
                mfcc_test, label_test = mfcc_test.to(DEVICE), label_test.to(DEVICE)
                # print(mfcc_test.shape)
                outputs = simpleARCNN(mfcc_test)
                _, predicted = torch.max(outputs.data, 1)
                total += label_test.size(0)
                acc += (predicted == label_test).sum()
                y_pred = (
                    predicted.data
                    if y_pred == None
                    else torch.cat((y_pred, predicted.data))
                )
                y_true = (
                    label_test
                    if y_true == None
                    else torch.cat((y_true, label_test))
                )
            acc = 100 * acc / total
            print(
                confusion_matrix(
                    y_true=y_true.cpu().detach().numpy(),
                    y_pred=y_pred.cpu().detach().numpy(),
                )
            )
            print(acc.item(), "%")


train()
