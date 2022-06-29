from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import numpy as np

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

scaler = StandardScaler()
X = mnist["data"]
X = scaler.fit_transform(X.astype(np.float32))
y = mnist["target"].astype(np.uint8)

X_train = X[:50000]
y_train = y[:50000]
X_test = X[50000:60000]
y_test = y[50000:60000]
X_val = X[60000:]
y_val = y[60000:]

X_train_sub = X[:5000]
y_train_sub = y[:5000]

# Extra Trees
# baseline
# ext_bl = ExtraTreesClassifier(random_state = 303)
# ext_bl.fit(X_train, y_train)
#
# # tuning
# ext = ExtraTreesClassifier(random_state = 303)
# params = {'n_estimators': [50, 100, 500, 1000],
#           'min_samples_split': [1, 2, 3, 4, 5]}
# ext_tune = GridSearchCV(ext, params, verbose = 2, cv = 3, n_jobs = 6)
# ext_tune.fit(X_train_sub, y_train_sub)
#
# print('Best ExtraTrees: ', ext_tune.best_estimator_)
# ext_tune.best_estimator_.fit(X_train, y_train)
# print('Tuned ExtraTrees valacc: ', ext_tune.best_estimator_.score(X_val, y_val))
# print('ExtraTrees baseline valacc: ', ext.score(X_val, y_val))
#
# ext_best = ExtraTreesClassifier(random_state = 303, n_estimators = 1000)
# ext_best.fit(X_train, y_train)

# Logistic Regression
# baseline
# lr_bl = LogisticRegression(random_state = 303, max_iter = 1000)
# lr_bl.fit(X_train, y_train)
#
# # tuning
# lr = LogisticRegression(random_state = 303, max_iter = 1000)
# params = {'C': [0.1, 0.5, 1.0, 3.0, 5.0],
#           'penalty': ['l1', 'l2', 'elasticnet']}
# lr_tune = GridSearchCV(lr, params, verbose = 2, cv = 3, n_jobs = 6)
# lr_tune.fit(X_train_sub, y_train_sub)
#
# print('Best LogReg: ', lr_tune.best_estimator_)
# lr_tune.best_estimator_.fit(X_train, y_train)
# print('Tuned LogReg valacc: ', lr_tune.best_estimator_.score(X_val, y_val))
# print('LogReg baseline valacc: ', lr_bl.score(X_val, y_val))

# lr_best = LogisticRegression(C = 0.1, random_state = 303)
# lr_best.fit(X_train, y_train)

# Random Forest
# baseline
# rf_bl = RandomForestClassifier(random_state = 303)
# rf_bl.fit(X_train, y_train)
#
# # tuning
# rf = RandomForestClassifier(random_state = 303, n_jobs = 6)
# params = {'n_estimators': [100, 500, 1000],
#           'max_depth': [10, 25, 50, 100],
#           'min_samples_leaf': [1, 2, 5, 10]}
# rf_tune = GridSearchCV(rf, params, verbose = 2, cv = 3, n_jobs = 6)
# rf_tune.fit(X_train_sub, y_train_sub)
#
# print('Best RF: ', rf_tune.best_estimator_)
# rf_tune.best_estimator_.fit(X_train, y_train)
# print('Tuned RF valacc: ', rf_tune.best_estimator_.score(X_val, y_val))
# print('RF baseline valacc: ', rf_bl.score(X_val, y_val))
#
# rf_best = RandomForestClassifier(random_state = 303, max_depth = 50,
#                                  n_estimators = 1000, n_jobs = 6)
# rf_best.fit(X_train, y_train)

# MLP
# baseline
# mlp_bl = MLPClassifier(random_state = 303)
# mlp_bl.fit(X_train, y_train)
#
# # tuning
# mlp = MLPClassifier(random_state = 303, max_iter = 1000)
# params = {'alpha': [.0001, .001, .01],
#           'learning_rate': ['constant', 'invscaling', 'adaptive'],
#           'learning_rate_init': [.001, .01, .1]}
# mlp_tune = GridSearchCV(mlp, params, verbose = 2, cv = 3)
# mlp_tune.fit(X_train_sub, y_train_sub)
#
# print('Best RF: ', mlp_tune.best_estimator_)
# mlp_tune.best_estimator_.fit(X_train, y_train)
# print('Tuned RF valacc: ', mlp_tune.best_estimator_.score(X_val, y_val))
# print('RF baseline valacc: ', mlp_bl.score(X_val, y_val))

# mlp_best = MLPClassifier(random_state = 302, max_iter = 1000)
# mlp_best.fit(X_train, y_train)

# Ensemble the best from above
# X_train = X_train_sub
# y_train = y_train_sub

ext_best = ExtraTreesClassifier(random_state = 303, n_estimators = 1000)
lr_best = LogisticRegression(C = 0.1, random_state = 303, max_iter = 1000)
rf_best = RandomForestClassifier(random_state = 303, max_depth = 50,
                                 n_estimators = 1000, n_jobs = 6)
mlp_best = MLPClassifier(random_state = 302, max_iter = 1000)

estimators = [('ExtraTrees', ext_best),
              #('LogReg', lr_best),
              ('RandomForest', rf_best),
              ('MLP', mlp_best)]
vc = VotingClassifier(estimators, voting = 'hard', n_jobs = 7)
vc.fit(X_train, y_train)

print('Validation Accuracies: ', [est.score(X_val, y_val) for est in vc.estimators_])
print('Test Accuracies: ', [est.score(X_test, y_test) for est in vc.estimators_])

print('VC Hard Val: ', vc.score(X_val, y_val))
print('VC Hard test: ', vc.score(X_test, y_test))

vc.voting = 'soft'
print('VC Soft Val: ', vc.score(X_val, y_val))
print('VC Soft test: ', vc.score(X_test, y_test))

# Blend
val_preds = np.empty((len(X_val), len(estimators)), dtype = np.float32)
tst_preds = np.empty((len(X_test), len(estimators)), dtype = np.float32)
for idx, estimator in enumerate(vc.estimators_):
    val_preds[:, idx] = estimator.predict(X_val)
    tst_preds[:, idx] = estimator.predict(X_test)

ext_blend = ExtraTreesClassifier(random_state = 42)
ext_blend.fit(val_preds, y_val)
print("Test blend accuracy: ", ext_blend.score(tst_preds, y_test))



