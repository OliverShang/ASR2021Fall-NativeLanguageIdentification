# Parameters
import torch.cuda
CLASS_NUM = 7
DEVICE = 'cuda:5' if torch.cuda.is_available() else 'cpu'