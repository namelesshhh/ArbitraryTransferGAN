from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 获得半月形的数据集
X, y = make_moons(n_samples=100, random_state=123)

# 建立目标维度为2的RBF模型
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)

# 使用KPCA降低数据维度，直接获得投影后的坐标
X_skernpca = scikit_kpca.fit_transform(X)

# 数据可视化
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()
