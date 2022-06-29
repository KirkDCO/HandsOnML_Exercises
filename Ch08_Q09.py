from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

import time

import numpy as np

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

scaler = StandardScaler()
X = mnist["data"]
X = scaler.fit_transform(X.astype(np.float32))
y = mnist["target"].astype(np.uint8)

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

rf_cls = RandomForestClassifier(random_state = 303, max_depth = 50,
                                n_estimators = 1000, n_jobs = 6)

start = time.time()
rf_cls.fit(X_train, y_train)
stop = time.time()

preds = rf_cls.predict(X_test)
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
print(stop - start)

pca = PCA(n_components = 0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

start = time.time()
rf_cls.fit(X_train_pca, y_train)
stop = time.time()

preds = rf_cls.predict(X_test_pca)
print(accuracy_score(y_test, preds))
print(confusion_matrix(y_test, preds))
print(stop - start)
