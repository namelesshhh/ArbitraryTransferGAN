import numpy as np
from scipy.cluster.vq import *
from pylab import *
from collections import defaultdict
import torch

class1 = torch.randn([4, 1000])
# class1 = pca.fit_transform(class0)
# print('class1.shape = '.format(class1.shape))
def k_means(class1, k):
    c,v = kmeans(class1, k)


    label = vq(class1, c)[0]

    labels = defaultdict(int)
    for i in label:
        labels[i] = labels[i] + 1

    labelsList = labels.values()
    newLabels = sorted(labelsList, reverse=True)
    sum_labels = sum(newLabels)

    classNum = 0
    tmpSum = 0
    for i in range(len(newLabels)):
        tmpSum = tmpSum + newLabels[i]
        if (tmpSum/ sum_labels > 0.8):
            classNum = i+1
            break

    #print('classNum = {}'.format(classNum))

    loss = (classNum - 1)**2

    #print('loss = {}'.format(loss))
    return loss
def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        # assign all pixels to the closest codebook element
        # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的ncluster类
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        # move each codebook element to be the mean of the pixels that assigned to it
        # 计算每一类的迭代中心，然后重新把第一轮随机选出的聚类中心移到这一类的中心处
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
    return c

if __name__ == "__main__":
    print("loss_kmeans = {}".format(kmeans(class1, 4, 1)))


