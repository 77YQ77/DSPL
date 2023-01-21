import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model import config
args = config.Arguments()
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

loss_fn = torch.nn.MSELoss(reduction='mean')## amis for weight_regularizer
def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def transfer_one2two(pred):
    aa = torch.zeros(pred.shape[0], 2)
    for i in range(len(pred)):
        if i >= 0.5:
            aa[i][1] = pred[i]
            aa[i][0] = 1 - pred[i]
        else:
            aa[i][1] = 1 - pred[i]
            aa[i][0] = pred[i]
    return aa

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.to(device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss

class SoftEntropy(nn.Module):
    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
    def forward(self, inputs, targets):
        # print("inputs",inputs.shape)
        # print("targets", targets.shape)
        log_probs = self.logsoftmax(inputs)
        # print('log_probs',log_probs)
        loss = (-targets.detach() * log_probs).mean(0).sum()
        # loss = self.nllloss_func(log_probs, targets)
        return loss

