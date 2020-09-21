import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Utils.utils as utils
import Utils.dpfuncdist as dputils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-8
def logzero(x):
    x = x + (x < EPS).float().detach() * EPS - (x > (1-EPS)).float().detach() * EPS
    return x

def class_balanced_bce_loss(outputs, labels, size_average=False, batch_average=True):
    assert(outputs.size() == labels.size())

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    loss_val = -(torch.mul(labels, torch.log(logzero(outputs))) + torch.mul((1.0 - labels), torch.log(logzero(1.0 - outputs))))

    loss_pos = torch.sum(torch.mul(labels, loss_val))
    loss_neg = torch.sum(torch.mul(1.0 - labels, loss_val))

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= torch.numel(labels)
    elif batch_average:
        final_loss /= labels.size()[0]
    return final_loss
