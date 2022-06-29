from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.base import clone
from scipy.stats import mode

import numpy as np

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

dt =  DecisionTreeClassifier(random_state = 303)
params = {'max_depth':  list(range(2, 16)),
          'min_samples_split': [2, 3, 4, 5, 10, 25, 50],
          'max_leaf_nodes': list(range(2, 100))}
dt_tune = GridSearchCV(dt, params, verbose = 2, cv = 3, n_jobs = 6)
dt_tune.fit(X_train, y_train)

print(dt_tune.best_estimator_)
print(dt_tune.best_score_)

preds = dt_tune.best_estimator_.predict(X_test)
print(accuracy_score(y_test, preds))

print(dt_tune.best_estimator_.get_depth(),
      dt_tune.best_estimator_.get_n_leaves())

# forests
mini_datasets = []
rs = ShuffleSplit(n_splits = 1000, test_size = 7900, random_state = 42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini = X_train[mini_train_index]
    y_mini = y_train[mini_train_index]
    mini_datasets.append((X_mini, y_mini))

forest = [clone(dt_tune.best_estimator_) for _ in range(1000)]
accuracy_scores = []

for tree, (X_mini, y_mini) in zip(forest, mini_datasets):
    tree.fit(X_mini, y_mini)
    preds = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, preds))

print(np.mean(accuracy_scores), np.min(accuracy_scores), np.max(accuracy_scores))

preds = np.empty([1000, len(X_test)], dtype = np.uint8)
for idx, tree in enumerate(forest):
    preds[idx] = tree.predict(X_test)

forest_preds, n_votes = mode(preds, axis = 0)

print(accuracy_score(y_test, forest_preds.reshape([-1])))