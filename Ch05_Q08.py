from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# use iris dataset - setosa vs versicolor are seperable
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train Linear SVC
C = 5

lsvc = LinearSVC(C = C, loss = 'hinge', random_state = 42)
svc = SVC(C = C, kernel = 'linear')
sgd = SGDClassifier(loss = 'hinge', random_state = 42, learning_rate = 'constant',
                    eta0 = 0.05)

lsvc.fit(X_scaled, y)
svc.fit(X_scaled,y)
sgd.fit(X_scaled,y)

print("LinearSVC:                   ", lsvc.intercept_ , lsvc.coef_)
print("SVC:                         ", svc.intercept_, svc.coef_)
print("SGDClassifier(alpha={:.5f}):".format(sgd.alpha), sgd.intercept_, sgd.coef_)

# Compute the slope and bias of each decision boundary
w1 = -lsvc.coef_[0, 0]/lsvc.coef_[0, 1]
b1 = -lsvc.intercept_[0]/lsvc.coef_[0, 1]
w2 = -svc.coef_[0, 0]/svc.coef_[0, 1]
b2 = -svc.intercept_[0]/svc.coef_[0, 1]
w3 = -sgd.coef_[0, 0]/sgd.coef_[0, 1]
b3 = -sgd.intercept_[0]/sgd.coef_[0, 1]

# Transform the decision boundary lines back to the original scale
line1 = scaler.inverse_transform([[-10, -10 * w1 + b1], [10, 10 * w1 + b1]])
line2 = scaler.inverse_transform([[-10, -10 * w2 + b2], [10, 10 * w2 + b2]])
line3 = scaler.inverse_transform([[-10, -10 * w3 + b3], [10, 10 * w3 + b3]])

# Plot all three decision boundaries
plt.figure(figsize=(11, 4))
plt.plot(line1[:, 0], line1[:, 1], "k:", label="LinearSVC")
plt.plot(line2[:, 0], line2[:, 1], "b--", linewidth=2, label="SVC")
plt.plot(line3[:, 0], line3[:, 1], "r-", label="SGDClassifier")
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs") # label="Iris versicolor"
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo") # label="Iris setosa"
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper center", fontsize=14)
plt.axis([0, 5.5, 0, 2])

plt.show()


