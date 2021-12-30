import torch.cuda
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy
from PIL import Image
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Parameters

CLASS_NUM = 6
KAGGLE_CLASS_NUM = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def scale_array(x, new_size):
    im = Image.fromarray(x)
    return np.array(im.resize((new_size[1], new_size[0]), Image.LINEAR))


def feature_reshape(feature, max_len=4001):
    x, y, z = feature.shape
    if z >= max_len:
        return feature[:, :, :max_len]
    else:
        feat_ = torch.zeros((x, y, max_len))
        feat_[:, :, :z] = feature
        return feat_


def feature_normalization(feature):

    minMaxScaler = MinMaxScaler()
    return minMaxScaler.fit_transform(feature)

def str2ascii(s):
    L = []
    for i in s:
        if int(ord(i)-0)>=ord('a') and int(ord(i)-0)<=ord('z'):
            L.append(int(ord(i)-ord('a'))+1)
        if int(ord(i)-0)>=ord('A') and int(ord(i)-0)<=ord('Z'):
            L.append(int(ord(i)-ord('A'))+1)
    print(L)
    return L

def ctcvec(x,leng):
    tmp = np.zeros((leng))
    #print("TMPL",tmp.shape)
    n = len(x)
    if n>leng:
        tmp = x[:leng]
    else:
        tmp[:n] = x
    #print("TMPL",tmp.shape)
    return tmp 

def ctcdict_kaggle():
    s = "Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
    s = s.lower()
    f = s.split()
    dct = {}
    revdct = {}
    p = 0
    for i in range(len(f)):
        if f[i] not in dct:
            dct[f[i]] = p
            revdct[p]=f[i]
            p = p+1
    lst = []
    for i in range(len(f)):
        lst.append(dct[f[i]])
    return dct,lst,revdct
