# Cuda
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Libraries
import nni
import torch
from torch.cuda import device
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchsummary import summary

# Utilities
from dataset import CommonVoiceDataset, KaggleDataset
from utils import DEVICE

# Models
from main_aesrc_model import SimpleAESRC
from resnet_official import ResNetCNN
from simple_cnn_classifier import SimpleARCNN, TinyCNN

# Losses
from loss_metrics import SphereProduct, FocalLoss, AddMarginProduct, ArcMarginProduct

# Configurations
batch_size = 8  # Batchsize
closs = "cross_entropy"  # Loss metrics
model = "simplecnn"  # Model
optm = "Adam"  # Optimizers
feat = "pre_calculated_mfcc"  # Input feature
faceloss = False  # When the value is True, the model will output features instead of logits or probabilities
segment_length = 1001  # Length for audio clips
total_epochs = 200  # Total epochs
final_actv = "none"  # Softmax or sphereface, etc
lr = 1e-4  # Learning rate
classes = 4  # Classes
hidden = 256  # Hidden size

# Dataset
train_set = KaggleDataset(
    mode="train", feature=feat, classes=classes, segment_length=segment_length
)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = KaggleDataset(
    mode="test", feature=feat, classes=classes, segment_length=segment_length
)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


if closs == "focalloss":
    asr_loss = FocalLoss().to(DEVICE)
elif closs == "cross_entropy":
    asr_loss = torch.nn.CrossEntropyLoss().to(DEVICE)


if final_actv == "sphereface":
    faceloss = True
    metrics_fc = SphereProduct(128, classes).to(DEVICE)
elif final_actv == "arcface":
    faceloss = True
    metrics_fc = ArcMarginProduct(128, classes).to(DEVICE)

if model == "simplecnn":
    simpleARCNN = SimpleARCNN(classes, 256, 128, faceloss).to(DEVICE)
elif model == "aesrc":
    simpleARCNN = SimpleAESRC(classes, 224, 256, 128, faceloss).to(DEVICE)
elif model == "tinycnn":
    simpleARCNN = TinyCNN(classes, 819264, 128, faceloss).to(DEVICE)
else:
    raise

if optm == "Adam":
    optimizer = torch.optim.Adam(simpleARCNN.parameters(), lr=lr)
else:
    optimizer = torch.optim.SGD(simpleARCNN.parameters(), lr=lr)


def train():
    # Summary
    summary(simpleARCNN, train_set[0][0].shape, device="cuda")

    # Train
    for epoch in range(total_epochs):
        print("\n")
        print("*********Epoch ", epoch, " of ", total_epochs, "************")
        with tqdm.tqdm(total=int(len(train_set) / batch_size), desc="Training") as t:
            simpleARCNN.train()
            total_loss = 0
            total_iter = 0
            total_acc = 0
            for i, (mfcc, label) in enumerate(train_loader):
                mfcc_gpu = mfcc.to(DEVICE)
                label_gpu = label.to(DEVICE)
                optimizer.zero_grad()
                if faceloss:
                    feature = simpleARCNN(mfcc_gpu)
                    outputs = metrics_fc(feature, label_gpu.argmax(1))
                else:
                    outputs = simpleARCNN(mfcc_gpu)
                loss = asr_loss(outputs, label_gpu)
                total_loss += loss.cpu().detach().numpy()
                _, predicted = torch.max(outputs.data, 1)
                total_acc += (predicted == label_gpu.argmax(1)).sum()
                loss.backward()
                optimizer.step()
                total_iter = total_iter + batch_size
                t.set_postfix(
                    loss=total_loss / total_iter,
                    acc=total_acc.cpu().numpy() / total_iter,
                )
                t.update(1)
        with torch.no_grad():
            simpleARCNN.eval()
            acc = 0
            total = 0
            y_true = None
            y_pred = None
            for test_data in test_loader:
                mfcc_test, label_test = test_data
                mfcc_test, label_test = mfcc_test.to(DEVICE), label_test.to(DEVICE)
                # print(mfcc_test.shape)
                if faceloss:
                    feature = simpleARCNN(mfcc_test)
                    outputs = metrics_fc(feature, label_test.argmax(1))
                else:
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
            print(
                confusion_matrix(
                    y_true=y_true.cpu().detach().numpy(),
                    y_pred=y_pred.cpu().detach().numpy(),
                )
            )
            print("Val Acc ", acc.item(), "%")


train()
