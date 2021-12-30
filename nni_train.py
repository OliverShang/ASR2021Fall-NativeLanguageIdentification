# Cuda
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

# Libraries
import nni
import logging
import torch
from torch.cuda import device
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchsummary import summary
from model.ensemble_net.ensemble_net import *
import sys
# Utilities
from utils.dataset import CommonVoiceDataset, KaggleDataset
from utils.utils import *

# Models
from model.crnn_net.main_crnn_model import CRNNet
from model.simplecnn.simple_cnn_classifier import SimpleARCNN, TinyCNN

# Losses
from utils.loss_metrics import SphereProduct, FocalLoss, AddMarginProduct, ArcMarginProduct

_logger = logging.getLogger("automl")
# Configurations
# batch_size = 8  # Batchsize
# closs = "cross_entropy"  # Loss metrics
# model = "simplecnn"  # Model
# optm = "Adam"  # Optimizers
feat = "mel_spectrogram"  # Input feature
faceloss = False  # When the value is True, the model will output features instead of logits or probabilities
segment_length = 1001  # Length for audio clips
total_epochs = 70  # Total epochs
# final_actv = "none"  # Softmax or sphereface, etc
# lr = 1e-4  # Learning rate
classes = 8  # Classes
# hidden = 256  # Hidden size
enable_ctc = True  # Enable CTC
enable_ctc_ = False  # Enable CTC (Do Not Configure)
ctclen = 353  # CTC Target Length
ctcinp = 252  # CTC Input Length
ctc_pretrain_epoch = 30  # CTC Pretrain Epoch
use_data = "kaggle"  # Dataset
ctc_weight = 0.0001
best_acc = 0.0  # Best accuracy
lossctc = torch.nn.CTCLoss(0, zero_infinity=True)

def prepare(args):
    global train_set
    global train_loader
    global test_set
    global test_loader
    global asr_loss
    global metrics_fc
    global simpleARCNN
    global optimizer
    global batch_size
    global faceloss
    global feat

    # Configuration
    lr = args["lr"]
    hidden = args["hidden"]
    batch_size = args["batch_size"]

    # Dataset
    train_set = KaggleDataset(
        mode="train", feature=feat, classes=classes, segment_length=segment_length
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = KaggleDataset(
        mode="test", feature=feat, classes=classes, segment_length=segment_length
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    if args["closs"] == "focalloss":
        asr_loss = FocalLoss().to(DEVICE)
    elif args["closs"] == "cross_entropy":
        asr_loss = torch.nn.CrossEntropyLoss().to(DEVICE)

    if args["final_actv"] == "sphereface":
        faceloss = True
        metrics_fc = SphereProduct(128, classes).to(DEVICE)
    elif args["final_actv"] == "arcface":
        faceloss = True
        metrics_fc = ArcMarginProduct(128, classes).to(DEVICE)

    if args["model"] == "simplecnn":
        simpleARCNN = SimpleARCNN(
            classes, conv_out=256, fc_ne=1024, feature_only=faceloss
        ).to(DEVICE)
    elif args["model"] == "net":
        simpleARCNN = CRNNet(
            classes, conv_output=64, hidden=hidden, out_feat=128, feature_only=faceloss
        ).to(DEVICE)
    elif args["model"] == "tinycnn":
        simpleARCNN = TinyCNN(
            accent_classes=classes,
            conv_output=160640,
            hidden=hidden,
            feature_only=faceloss,
        ).to(DEVICE)
    elif args["model"] == "ensemble":
        simpleARCNN = EnsembleNet2(classes,256,64, hidden).to(DEVICE,)
    else:
        raise

    if args["optm"] == "Adam":
        optimizer = torch.optim.Adam(simpleARCNN.parameters(), lr=lr)
    elif args["optm"] == "Adadelta":
        optimizer = torch.optim.Adadelta(simpleARCNN.parameters(), lr=lr)
    elif args["optm"] == "Adagrad":
        optimizer = torch.optim.Adagrad(simpleARCNN.parameters(), lr=lr)
    elif args["optm"] == "Adamax":
        optimizer = torch.optim.Adamax(simpleARCNN.parameters(), lr=lr)
    elif args["optm"] == "SGD":
        optimizer = torch.optim.SGD(simpleARCNN.parameters(), lr=lr)
    else:
        raise
    # Summary
    # summary(simpleARCNN, train_set[0][0].shape, device="cuda")



def train(epoch):
    # Train
    with tqdm.tqdm(total=int(len(train_set) / batch_size), desc="Training",file=sys.stdout) as t:
        simpleARCNN.train()
        total_loss = 0
        total_lossmx = 0
        total_lossctc = 0
        total_iter = 0
        total_acc = 0
        if not enable_ctc_:
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
        else:
            for i, (mfcc, label, ctc_lbl, ctc_lbllen, ctcl) in enumerate(train_loader):
                mfcc_gpu = mfcc.to(DEVICE)
                label_gpu = label.to(DEVICE)
                ctclb_gpu = ctc_lbl
                ctclblen_gpu = ctc_lbllen
                ctcl_gpu = ctcl
                # print(ctcl_gpu)
                # print(fw.numpy())
                optimizer.zero_grad()
                if faceloss:
                    feature, ctcout = simpleARCNN(mfcc_gpu)
                    outputs = metrics_fc(feature, label_gpu.argmax(1))
                else:
                    outputs, ctcout = simpleARCNN(mfcc_gpu)
                    # print(ctcout)
                '''
                print("ARR",ctcout.shape)
                ctcoutputcpu = ctcout.detach().cpu().permute(1,0,2)
                for i in range(ctcoutputcpu.shape[0]):
                    stp = ""
                    for j in range(ctcoutputcpu.shape[1]):
                        stp+=chr(int(torch.argmax(ctcoutputcpu[i,j]))+ord('A')-1)
                    print("BATCH_PRED",stp)
                for i in range(1):
                    stp = ""
                    for j in range(ctclb_gpu.shape[1]):
                        stp+=chr(int(ctclb_gpu[i,j].detach().item())+ord('A')-1)
                    print("STANDARD_ANSW",stp)
                '''
                loss = asr_loss(outputs, label_gpu)
                loss2 = lossctc(ctcout, ctclb_gpu, ctcl_gpu, ctclblen_gpu)
                if epoch < ctc_pretrain_epoch:
                    mixed_loss = loss2
                else:
                    if torch.isnan(loss2.detach()):
                        mixed_loss = loss
                    else:
                        mixed_loss = loss + loss2*0.001
                total_loss += loss.cpu().detach().numpy()
                total_lossctc += loss2.cpu().detach().numpy()
                total_lossmx += mixed_loss.cpu().detach().numpy()
                _, predicted = torch.max(outputs.data, 1)
                total_acc += (predicted == label_gpu.argmax(1)).sum()
                mixed_loss.backward()
                if epoch < ctc_pretrain_epoch:
                    optimizerpr.step()
                else:
                    optimizer.step()
                total_iter = total_iter + batch_size
                t.set_postfix(
                    class_loss=total_loss / total_iter,
                    acc=total_acc.cpu().numpy() / total_iter,
                    ctc_loss=total_lossctc / total_iter,
                    mixed_loss=total_lossmx/total_iter
                )
                t.update(1)
    return total_acc.cpu().numpy() / total_iter, total_loss / total_iter


def test():
    global simpleARCNN
    global test_loader
    global test_set
    global best_acc

    with torch.no_grad():
        simpleARCNN.eval()
        acc = 0
        total = 0
        y_true = None
        y_pred = None
        ctcresult=[]
        for test_data in tqdm.tqdm(test_loader,desc="Testing "):

            # print(mfcc_test.shape)
            if enable_ctc_:
                if faceloss:
                    mfcc_test, label_test, _a, _b, _c = test_data
                    mfcc_test, label_test = (
                        mfcc_test.to(DEVICE,),
                        label_test.to(DEVICE,),
                    )
                    feature, ctcoutput = simpleARCNN(mfcc_test)
                    outputs = metrics_fc(feature, label_test.argmax(1))
                else:
                    mfcc_test, label_test, _a, _b, _c = test_data
                    mfcc_test, label_test = (
                        mfcc_test.to(DEVICE,),
                        label_test.to(DEVICE,),
                    )
                    outputs, ctcoutput = simpleARCNN(mfcc_test)
                ctcoutputcpu = ctcoutput.detach().cpu().permute(1,0,2)
                for i in range(ctcoutputcpu.shape[0]):
                    stp = ""
                    _,_,revd = ctcdict_kaggle()
                    last = ""
                    for j in range(ctcoutputcpu.shape[1]):
                        if revd[int(torch.argmax(ctcoutputcpu[i,j]))]!= last:
                            last = revd[int(torch.argmax(ctcoutputcpu[i,j]))]
                            stp+=revd[int(torch.argmax(ctcoutputcpu[i,j]))]+" "
                    ctcresult.append(stp+" ")
                
            else:
                mfcc_test, label_test = test_data
                mfcc_test, label_test = mfcc_test.to(DEVICE,), label_test.to(DEVICE,)
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
        for i in ctcresult:
            print(i)
        
        print("Val Acc ", acc.item(), "%")
        if acc.item() > best_acc:
            best_acc = acc.item()

        return acc.item(), best_acc


if __name__ == "__main__":

    try:
        RCV_CONFIG = nni.get_next_parameter()

        # RCV_CONFIG = {'lr': 0.1, 'optimizer': 'Adam', 'model':'senet18'}
        _logger.debug(RCV_CONFIG)
        prepare(RCV_CONFIG)
        global optimizerpr
        optimizerpr = torch.optim.Adam(simpleARCNN.parameters(), lr=1e-4)
        acc = 0.0
        best_acc = 0.0
        for epoch in range(total_epochs):
            print("\n")
            print("*********Epoch ", epoch, " of ", total_epochs, "************")
            loss = train(epoch)
            acc, best_acc = test()
            nni.report_intermediate_result(acc)
            nni.training_update(loss)

        nni.report_final_result(best_acc)
    except Exception as exception:
        _logger.exception(exception)
        raise
