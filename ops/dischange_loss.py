import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from clearml import Task, Logger
from ops.histogram_matching import histogram_matching



class DischangeLoss(nn.Module):
    def __init__(self):
        super(DischangeLoss, self).__init__()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def forward(self, input_data, makeup_data,mask_src):

        input_data = (self.de_norm(input_data) * 255).squeeze()
        makeup_data = (self.de_norm(makeup_data) * 255).squeeze()

        mask_src = mask_src.expand(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        # print("input_data:", input_data.shape)
        # print("mask_src:",mask_src.shape)
        input_masked = input_data * mask_src
        target_masked = makeup_data * mask_src

        loss = F.l1_loss(input_masked, target_masked)

        return loss


