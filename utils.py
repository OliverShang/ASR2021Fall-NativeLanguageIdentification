import torch.cuda

# Parameters

CLASS_NUM = 6
KAGGLE_CLASS_NUM = 5
DEVICE = "cuda:4" if torch.cuda.is_available() else "cpu"

