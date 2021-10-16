# -*- coding:utf-8 -*-

"""
The attentional mechanism is used for active feature fusion.
https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
"""
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)


        output = self.out(concat)

        return output
"""
def active_feature_fusion(content_feature, style_feature, embed_dim, num_heads = 10, ):
    """
    :param content_feature: (B_c, L, trasnformer output size)
    :param style_feature: (B_s, L, trasnformer output size)
    :param embed_dim: The same as trasnformer output size
    :param num_heads: multihead attention head numbers
    :return: The features obtained by the active fusion of content features and style features
    """
    B_s = style_feature.size()[0] #Batch Size
    B_c = content_feature.size()[0]
    multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    list_attn = []
    for i in range(B_c):
        content = content_feature[i]
        content = content[None, :, :]
        query = content.repeat(B_s + 1, 1, 1)
        key = value = torch.cat((content, style_feature), 0)
        attn_output, attn_output_weights = multihead_attn(query, key, value)
        attn_output = torch.sum(attn_output, 0)
        #print("fusion_feature = {} | attn_output = {}".format(fusion_feature.size(), attn_output.size()))
        list_attn.append(attn_output)
    fusion_feature = torch.stack(list_attn, 0)

    return fusion_feature

if __name__ == "__main__":
    content_feature = torch.randn([16, 49,1024] , requires_grad=True)
    style_feature = torch.randn([16, 49, 1024] , requires_grad=True)
    out = active_feature_fusion(content_feature, style_feature, style_feature.size()[-1], 8)
    print(out.size())


