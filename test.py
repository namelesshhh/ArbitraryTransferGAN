# -*- coding:utf-8 -*-
import torch
t = torch.tensor([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])

print(t)
t = t.flatten(start_dim=1)
print(t)
t = t.resize(*[2,2,2])

print(t)