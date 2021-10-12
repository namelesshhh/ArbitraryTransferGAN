import numpy as np
from pylab import *
from collections import defaultdict
import torch
from functools import wraps
class1 = torch.randn([64, 1000] ,dtype=float, requires_grad=True)
#class1 = torch.ones([64, 5])
# class1 = pca.fit_transform(class0)
# print('class1.shape = '.format(class1.shape))

def decorator_kmeans(func):

    @wraps(func)
    def wrapFunc(x, ncluster, niter=10):
        c, label = func(x, ncluster, niter)
        #add content
        labels = defaultdict(int)
        for i in label:
            i = int(i)
            labels[i] = labels[i] + 1


        newLabels = sorted(labels.items(), key = lambda d:d[1], reverse=True)

        # sum_labels = sum(newLabels)
        #
        # classNum = 0
        # tmpSum = 0
        # for i in range(len(newLabels)):
        #     tmpSum = tmpSum + newLabels[i]
        #     if (tmpSum / sum_labels > 0.8):
        #         classNum = i + 1
        #         break
        #
        # #print('classNum = {}'.format(classNum))
        #
        # loss = (classNum - 1) ** 2
        #print("label.size = {}".format(label.size()))
        #print('newLabels[0][0] = {}'.format(newLabels[0][0]))
        #TrueLabels = torch.tensor(newLabels[0][0],dtype=float, requires_grad=True)
        #print("TrueLabels = {}".format(TrueLabels))
        CenterTensor = c[newLabels[0][0]]
        #print("Center tensor = {}".format(CenterTensor))
        Center = CenterTensor.repeat(x.size()[0], 1)
        #print("Center size = {}".format(Center.size()))
        return label, Center

    return wrapFunc

@decorator_kmeans
def kmeans(x, ncluster, niter=10):
    """
    Find the biggest bunch， and view it as truth label
    :param x: torch.tensor(data_num,data_dim)
    :param ncluster: The number of clustering for data_num
    :param niter: Number of iterations for kmeans
    :return:label of data, and turth data center that is produced by the biggest bunch
    """

    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    #print("c.size = {}".format(c[None, :, :].size()))
    #print("x.size = {}".format(x[:, None, :].size()))
    m = x[:, None, :] - c[None, :, :]
    #print("m.size = {}".format(m.sum(-1).argmin(1).size()))
    for i in range(niter):
        # assign all pixels to the closest codebook element
        # .argmin(1) : 按列取最小值的下标,下面这行的意思是将x.size(0)个数据点归类到random选出的ncluster类
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1).double()
        # move each codebook element to be the mean of the pixels that assigned to it
        # 计算每一类的迭代中心，然后重新把第一轮随机选出的聚类中心移到这一类的中心处
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        # re-assign any poorly positioned codebook elements
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        #print('done step %d/%d, re-initialized %d dead clusters' % (i+1, niter, ndead))
        c[nanix] = x[torch.randperm(N)[:ndead]] # re-init dead clusters
        #print("a.size = {}".format(a.size()))
    #print("After: c.size = {}".format(c.size()))
    return c, a

if __name__ == "__main__":
    import torch.nn as nn


    input, target = kmeans(class1, 5, 1)
    print("tyype = {}".format(target.dtype))
    #print("mode = {}".format(kmeans(class1, 5, 1)))

    loss = nn.MSELoss()
    output = loss(class1, target)
    output.backward()
    print("loss = {}".format(output))



