from torchsummary import summary
from model.ensemble_net.ensemble_net import *
import tqdm
from utils.dataset import KaggleDataset
from torch.utils.data import DataLoader
import torch
import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torchsummary import summary
import numpy as np
import itertools
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.preprocessing import label_binarize
model = torch.load("./checkpoints/mixed-1640818667.2191012/mixed-1640818667.2191012_7_63.235294342041016.pth")
test_set = KaggleDataset(
    mode="test",
    feature="mel_spectrogram",
    classes=8,
    segment_length=1001,
    enable_ctc=False,
    ctclen=252,
)
test_loader = DataLoader(test_set, batch_size=16, shuffle=True)



def rocCurve(probs, labels):
    y_test = label_binarize(labels, classes=[0, 1, 2, 3, 4, 5, 6, 7])
    plt.figure()
    plt.title("ROC CURVE")
    region = ["arabic", "china-cantonese", "china-mandarin", "dutch", "english", "french", "korean", "russian"]
    for i in range(8):
        # 计算每个类别的FPR, TPR
        fpr, tpr, thr = roc_curve(y_test[:, i], probs[:, i])
        plt.plot(fpr, tpr, linestyle='--', label="{},AUC: {:.2f}".format(region[i], auc(fpr, tpr)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    fpr, tpr, thresholds = roc_curve(y_test.ravel(), probs.ravel())
    print(roc_auc_score(y_test, probs, average='micro'))
    plt.plot(fpr, tpr, linestyle="--", label="average,AUC: {:.2f}".format(roc_auc_score(y_test, probs, average='micro')))
    plt.legend(loc="lower right")
    plt.savefig('roc.png')
    print("AUC:", roc_auc_score(y_test, probs, multi_class="ovr", average=None))

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds,
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("confusion_matrix.png")




def test():
    with torch.no_grad():
        model.eval()
        acc = 0
        total = 0
        y_true = None
        y_pred = None
        prob = None
        ctcresult=[]
        for test_data in tqdm.tqdm(test_loader,desc="Testing "):
            mfcc_test, label_test = test_data
            mfcc_test, label_test = mfcc_test.to(DEVICE,), label_test.to(DEVICE,)
            outputs = model(mfcc_test)
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
            prob = (
                outputs.data
                if prob == None
                else torch.cat((prob, outputs.data))
            )
        rocCurve(prob.cpu().detach().numpy(), y_true.cpu().detach().numpy())
        acc = 100 * acc / total
        cm = confusion_matrix(
                y_true=y_true.cpu().detach().numpy(),
                y_pred=y_pred.cpu().detach().numpy(),
            )
        for i in ctcresult:
            print(i)
        print("Val Acc ", acc.item(), "%")
        plot_confusion_matrix(cm, ["arabic", "china-cantonese", "china-mandarin", "dutch", "english", "french", "korean", "russian"])
        return cm

if __name__ == '__main__':
    test()