import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torchvision import datasets

import trailnet


NUM_EPOCHS = 20


# Loss layer for cross entropy with softmax and entropy optimization.
class CrossEntropySoftmaxWithEntropyLossLayer(Function):

    def __init__(self, ent_scale=0.01, p_scale=0.0001, label_eps=0.0):
        self.ent_scale = ent_scale
        self.p_scale = p_scale
        self.label_eps = label_eps

    def forward(self, bottom, top):
        lgt_blob = bottom[0].data
        lab_blob = bottom[1].data
        res_blob = top[0].data
        total_loss = 0.0
        smooth_lab  = np.zeros(3)
        smooth_val  = self.label_eps / (len(smooth_lab) - 1)
        for i in range(bottom[0].data.shape[0]):
            lab = int(lab_blob[i])
            lgt = lgt_blob[i]
            sm   = self.softmax(lgt)
            lse  = self.log_sum_exp(lgt)
            smooth_lab.fill(smooth_val)
            smooth_lab[lab] = 1.0 - self.label_eps
            ce   = -np.sum(smooth_lab * (lgt - lse))
            ent  = -np.sum(sm * (lgt - lse))
            loss = ce - self.ent_scale * ent
            scale = [self.p_scale, 0.0, self.p_scale]
            loss += scale[lab] * sm[2 - lab]
            total_loss += loss
        res_blob[0] = total_loss / bottom[0].data.shape[0]

    def backward(self, top, propagate_down, bottom):
        lgt_blob = bottom[0].data
        lab_blob = bottom[1].data
        lgt_diff = bottom[0].diff
        smooth_lab  = np.zeros(3)
        smooth_val  = self.label_eps / (len(smooth_lab) - 1)
        for i in range(bottom[0].data.shape[0]):
            lab = int(lab_blob[i])
            lgt = lgt_blob[i]
            sm  = self.softmax(lgt)
            log_sm = self.log_softmax(lgt)
            smooth_lab.fill(smooth_val)
            smooth_lab[lab] = 1.0 - self.label_eps
            a = np.sum((1.0 + log_sm) * sm) - 1.0
            ent_diff    = sm * (a - log_sm)
            lgt_diff[i] = (sm - smooth_lab) - self.ent_scale * ent_diff
            scale                = [self.p_scale, 0.0, self.p_scale]
            lgt_diff             -= scale[lab] * sm[2 - lab] * sm
            lgt_diff[i, 2 - lab] += scale[2 - lab] * sm[2 - lab]
        lgt_diff /= bottom[0].data.shape[0]

    def softmax(self, logits):
        e = np.exp(logits - np.max(logits))
        return e / np.sum(e)

    def log_sum_exp(self, x):
        a = np.max(x)
        return a + np.log(np.sum(np.exp(x - a)))

    def log_softmax(self, logits):
        return logits - self.log_sum_exp(logits)


def main():
    office_path_dataset = datasets.ImageFolder(root=sys.argv[1])
    train_data = torch.utils.data.DataLoader(office_path_dataset, batch_size=64, shuffle=True, num_workers=4)

    net = trailnet.get_trailnet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95, nesterov=True)

    for epoch in range(NUM_EPOCHS):
        print(epoch)


if __name__ == '__main__':
    main()
