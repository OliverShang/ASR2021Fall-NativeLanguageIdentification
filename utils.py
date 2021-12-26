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
KAGGLE_CLASS_NUM = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def scale_array(x, new_size):
    im = Image.fromarray(x)
    return np.array(im.resize((new_size[1],new_size[0]), Image.LINEAR))

def feature_reshape(feature, max_len=1001):
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

