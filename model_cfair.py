
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)



class cfair(nn.Module):
    def __init__(self):
        super.__init__()
        

    def forward(self, x):
        return out

class Classifier(nn.Module):
    def __init__(self, in_dim = 114, hid_dim = 60, include_A = False):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(in_dim+1, hid_dim) if include_A else nn.Linear(in_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.cls = nn.Linear(hid_dim, 2)
        self.relu = nn.ReLU()
        
    def forward(self, input):
        #hidden = self.relu(self.layer1(input))
        hidden = self.relu(self.layer1(input))
        # hidden = self.relu(self.layer2(hidden))
        out = self.cls(hidden)
        logits = F.log_softmax(out, dim=1)
        return hidden, logits

class Adversary(nn.Module):
    def __init__(self, in_dim = 60, hid_dim = 50):
        super(Adversary, self).__init__()
        self.layer1 = nn.Linear(in_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.cls = nn.Linear(hid_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, input):
        #out = self.relu(self.layer1(input))
        # conditional_idx = (self.y==label).squeeze()
        # input = input[conditional_idx]
        out = self.relu(self.layer1(input))
        out = self.relu(self.layer2(out))
        out = self.cls(out)
        logits = F.log_softmax(out, dim=1)
        return logits

class TransferNet(nn.Module):
    def __init__(self, in_dim, include_A = False):
        super(TransferNet, self).__init__()
        self.cls = nn.Linear(in_dim, 2)
    
    def forward(self, input):
        out = self.cls(input)
        logits = F.log_softmax(out, dim=1)
        return logits






