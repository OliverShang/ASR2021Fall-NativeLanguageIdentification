# Cuda
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"

# Libraries
import time
import nni
import logging
import torch
import tqdm
from sklearn.metrics import confusion_matrix
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchsummary import summary
from model.bigru_attention_net.attention_net import BiGruAttentionNet
from tensorboardX import SummaryWriter
import hiddenlayer as hl
# Utilities
from utils.dataset import CommonVoiceDataset, KaggleDataset
from utils.utils import *

# Models
from model.crnn_net.main_crnn_model import CRNNet
from model.simplecnn.simple_cnn_classifier import SimpleARCNN, TinyCNN
from model.ensemble_net.ensemble_net import EnsembleNet2

# Losses
from utils.loss_metrics import (
    SphereProduct,
    FocalLoss,
    AddMarginProduct,
    ArcMarginProduct,
)

_logger = logging.getLogger("automl")
# Configurations
batch_size = 16  # Batchsize
closs = "cross_entropy"  # Loss metrics
model = "mixed"  # Model
optm = "Adamax"  # Optimizers
feat = "mel_spectrogram"  # Input feature
faceloss = False  # When the value is True, the model will output features instead of logits or probabilities
segment_length = 1001  # Length for audio clips
total_epochs = 80  # Total epochs
final_actv = "none"  # Softmax or sphereface, etc
lr = 0.0001  # Learning rate
lr_pr = 1e-3  # Learning rate for pretraining
classes = 8  # Classes
hidden = 64  # Hidden size
default_dtype = torch.double
best_acc = 0.0  # Best accuracy
enable_ctc = True  # Enable CTC
enable_ctc_ = False  # Enable CTC (Do Not Configure)
ctclen = 353  # CTC Target Length
ctcinp = 252  # CTC Input Length
ctc_pretrain_epoch = 30  # CTC Pretrain Epoch
use_data = "kaggle"  # Dataset
ctc_weight = 0.0001

config = {
    "batch_size":batch_size,
    "closs":closs,
    "model":model,
    "optm":optm,
    "feature":feat,
    "segment_length":segment_length,
    "total_epochs":total_epochs,
    "final_actv":final_actv,
    "lr":lr,
    "lr_pr":lr_pr,
    "classes":classes,
    "hidden":hidden,
    "enable_ctc_":enable_ctc_,
    "ctclen":ctclen,
    "ctcinp":ctcinp,
    "use_data":use_data,
    "ctc_pretrain_epoch":ctc_pretrain_epoch
}

if closs == "focalloss":
    asr_loss = FocalLoss().to(DEVICE,)
elif closs == "cross_entropy":
    asr_loss = torch.nn.CrossEntropyLoss().to(DEVICE,)
if final_actv == "sphereface":
    faceloss = True
    metrics_fc = SphereProduct(8, classes).to(DEVICE,)
elif final_actv == "arcface":
    faceloss = True
    metrics_fc = ArcMarginProduct(8, classes).to(DEVICE,)

if model == "simplecnn":
    train_model = SimpleARCNN(
        classes, conv_out=256, fc_ne=1024, feature_only=faceloss
    ).to(DEVICE,)
elif model == "crnn":
    if enable_ctc:
        enable_ctc_ = True
    train_model = CRNNet(
        classes,
        conv_output=64,
        hidden=hidden,
        out_feat=128,
        feature_only=faceloss,
        ctc_branch=enable_ctc,
        ctc_classes=55
    ).to(DEVICE,)
elif model == "tinycnn":
    train_model = TinyCNN(classes, 208832, hidden=hidden, feature_only=faceloss).to(
        DEVICE,
    )
elif model == "bigruattn":
    train_model = BiGruAttentionNet(classes).to(DEVICE,)
elif model == "mixed":
    train_model = EnsembleNet2(classes,256,64).to(DEVICE,)
else:
    raise


if optm == "Adam":
    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr)
elif optm == "Adadelta":
    optimizer = torch.optim.Adadelta(train_model.parameters(), lr=lr)
elif optm == "Adagrad":
    optimizer = torch.optim.Adagrad(train_model.parameters(), lr=lr)
elif optm == "Adamax":
    optimizer = torch.optim.Adamax(train_model.parameters(), lr=lr)
elif optm == "SGD":
    optimizer = torch.optim.SGD(train_model.parameters(), lr=lr)
else:
    raise

# CTC Loss
lossctc = torch.nn.CTCLoss(0, zero_infinity=True)
optimizerpr = torch.optim.Adam(train_model.parameters(), lr=lr_pr)

# Dataset
if use_data == "cv":
    train_set = CommonVoiceDataset(
        mode="train",
        feature=feat,
        classes=classes,
        segment_length=segment_length,
        enable_ctc=enable_ctc_,
        ctclen=ctcinp,
    )
else:
    train_set = KaggleDataset(
        mode="train",
        feature=feat,
        classes=classes,
        segment_length=segment_length,
        enable_ctc=enable_ctc_,
        ctclen=ctcinp,
    )

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
if use_data == "cv":
    test_set = CommonVoiceDataset(
        mode="test",
        feature=feat,
        classes=classes,
        segment_length=segment_length,
        enable_ctc=enable_ctc_,
        ctclen=ctcinp,
    )
else:
    test_set = KaggleDataset(
        mode="test",
        feature=feat,
        classes=classes,
        segment_length=segment_length,
        enable_ctc=enable_ctc_,
        ctclen=ctcinp,
    )
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Summary
summary(
    train_model, train_set[0][0].shape, device="cuda",
)

# Check Dataset
for i in tqdm.trange(11900,len(train_set)):
    train_set[i]

# Visualization
rn = test_set[0][0].shape
rx = torch.rand(2,rn[0],rn[1],rn[2])
with SummaryWriter(comment=model) as w:
    w.add_graph(train_model, rx) 
vis_graph = hl.build_graph(train_model, rx)
vis_graph.theme = hl.graph.THEMES["blue"].copy()
vis_graph.save("./model_vis.png")

def train(epoch):
    # Train
    with tqdm.tqdm(total=int(len(train_set) / batch_size), desc="Training",file=sys.stdout) as t:
        train_model.train()
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
                    feature = train_model(mfcc_gpu)
                    outputs = metrics_fc(feature, label_gpu.argmax(1))
                else:
                    outputs = train_model(mfcc_gpu)
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
                    feature, ctcout = train_model(mfcc_gpu)
                    outputs = metrics_fc(feature, label_gpu.argmax(1))
                else:
                    outputs, ctcout = train_model(mfcc_gpu)
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
    global train_model
    global test_loader
    global test_set
    global best_acc

    with torch.no_grad():
        train_model.eval()
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
                    feature, ctcoutput = train_model(mfcc_test)
                    outputs = metrics_fc(feature, label_test.argmax(1))
                else:
                    mfcc_test, label_test, _a, _b, _c = test_data
                    mfcc_test, label_test = (
                        mfcc_test.to(DEVICE,),
                        label_test.to(DEVICE,),
                    )
                    outputs, ctcoutput = train_model(mfcc_test)
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
                    feature = train_model(mfcc_test)
                    outputs = metrics_fc(feature, label_test.argmax(1))
                else:
                    outputs = train_model(mfcc_test)
            _, predicted = torch.max(outputs.data, 1)
            total += label_test.size(0)
            acc += (predicted == label_test.argmax(1)).sum()
            prob = (
                outputs.data
                if y_pred == None
                else torch.cat((prob, outputs.data))
            )
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
    acc = 0.0
    timestamp = str(time.time())
    os.mkdir("./checkpoints/"+model+"-"+timestamp)
    best_acc = 0.0
    train_acc_list = []
    test_acc_list = []
    for epoch in range(total_epochs):
        print("\n")
        print("*********Epoch ", epoch+1, " of ", total_epochs, "************")
        if epoch < ctc_pretrain_epoch and enable_ctc_:
            print("ASR-CTC Pretraining")
        else:
            print("Accent Recognition Training")
        train_acc,train_loss = train(epoch)
        acc, best_acc = test()
        test_acc_list.append(acc)
        train_acc_list.append(train_acc)
        print("Best Val Acc:", best_acc, "%")
        if acc>=best_acc:
            torch.save(train_model,"./checkpoints/"+model+"-"+timestamp+"/"+model+"-"+timestamp+"_"+str(epoch)+"_"+str(acc)+".pth")
    with open("./checkpoints/"+model+"-"+timestamp+"/trainlog.csv","w") as f:
        f.write(str(train_acc_list))
    with open("./checkpoints/"+model+"-"+timestamp+"/testlog.csv","w") as f:
        f.write(str(test_acc_list))
    with open("./checkpoints/"+model+"-"+timestamp+"/config.json","w") as f:
        f.write(str(config))

