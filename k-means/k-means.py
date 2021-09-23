import numpy as np
from scipy.cluster.vq import *
from pylab import *
from collections import defaultdict

class1 = 1.5 * np.random.randn(100, 256*256)
# class1 = pca.fit_transform(class0)
# print('class1.shape = '.format(class1.shape))

c,v = kmeans(class1, 5)


label = vq(class1, c)[0]

labels = defaultdict(int)
for i in label:
    labels[i] = labels[i] + 1

labelsList = labels.values()
newLabels = sorted(labelsList, reverse=True)
sum = sum(newLabels)

classNum = 0
tmpSum = 0
for i in range(len(newLabels)):
    tmpSum = tmpSum + newLabels[i]
    if (tmpSum/ sum > 0.8):
        classNum = i+1
        break

print('classNum = {}'.format(classNum))

loss = (classNum - 1)**2

print('loss = {}'.format(loss))