from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d

import numpy as np

from matplotlib import pyplot as plt

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

np.random.seed(42)
m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx].astype(np.uint8)

# tsne = TSNE(n_components=2, random_state=42)
# X_reduced = tsne.fit_transform(X)
#
# plt.figure(figsize=(13,10))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = y, cmap = "jet")
# plt.axis('off')
# plt.colorbar()
# plt.show()
#
tsne = TSNE(n_components=3, random_state=42)
X_reduced = tsne.fit_transform(X)

fig = plt.figure(figsize = (13, 10))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter3D(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
             c = y, cmap = 'jet')
plt.show()