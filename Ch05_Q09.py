from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

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

X_train_sub = X[:5000]
y_train_sub = y[:5000]

# randomized search hyperparameter tuning
svc = SVC()
params = {'gamma': reciprocal(0.001, 0.1),
          'C': uniform(1, 10)}
svc_tune = RandomizedSearchCV(svc, params, n_iter = 10, verbose = 2, cv = 3, n_jobs = 6)
svc_tune.fit(X_train_sub, y_train_sub)

print(svc_tune.best_estimator_)
print(svc_tune.best_score_)

svc_tune.best_estimator_.fit(X_train, y_train)
preds = svc_tune.best_estimator_.predict(X_test)
print(accuracy_score(y_test, preds))

