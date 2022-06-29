from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

dt =  DecisionTreeClassifier(random_state = 42)
params = {#'max_depth':  list(range(2, 16)),
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