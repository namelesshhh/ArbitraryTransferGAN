# -*- coding:utf-8 -*-
import torch
size_real = 128
real_label = 1
labels_real = torch.full((size_real,), real_label, dtype=torch.float)
labels_real = torch.squeeze(labels_real)
print("labels_real.size() = {}".format(labels_real.size()))