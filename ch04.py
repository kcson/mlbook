import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from matplotlib.colors import ListedColormap

# X = 2 * np.random.rand(100, 1)
# X_b = np.c_[np.ones((100, 1)), X]
# y = 4 + 3 * X + np.random.randn(100, 1)
#
# eta = 0.01
# n_iterations = 1000
# m = 100
#
# n_epochs = 50
# t0, t1 = 5, 50
#
#
# def learning_schedule(t):
#     return t0 / (t + t1)
#
#
# theta = np.random.randn(2, 1)
# for epoch in range(n_epochs):
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index:random_index + 1]
#         yi = y[random_index:random_index + 1]
#         gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
#         eta = learning_schedule(epoch * m + i)
#         theta = theta - eta * gradients
# for iteration in range(n_iterations):
#     gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
#     theta = theta - eta * gradients
# print(theta)
#
# x_p = [[0], [2]]
# x_p_b = np.c_[np.ones((2, 1)), x_p]
# predict = x_p_b.dot(theta)
#
# plt.plot(x_p, predict, "r-")
# plt.plot(X, y, "b.")
# plt.show()
#
# sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.01)
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.intercept_, sgd_reg.coef_)
# print(X_b)
#
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)
#
# X_new = np.array([[0], [2]])
# X_new_b = np.c_[np.ones((2, 1)), X_new]
# y_predict = X_new_b.dot(theta_best)
# print(y_predict)
#
# # plt.plot(X_new, y_predict, "r-")
# # plt.plot(X, y, "b.")
# # plt.show()
#
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# print(lin_reg.intercept_, lin_reg.coef_)
# print(lin_reg.predict(X_new))
m = 100
X = 6 * np.random.rand(m, 1) - 3
# print(X)
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# print(y)
# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly_features.fit_transform(X)

# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y)
# print(lin_reg.intercept_, lin_reg.coef_)

# X_new = np.linspace(-3, 3, 100).reshape(100, 1)
# X_new_poly = poly_features.transform(X_new)
# y_new = lin_reg.predict(X_new_poly)

# for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
#     polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
#     std_scaler = StandardScaler()
#     lin_reg = LinearRegression()
#     polynomial_reg = Pipeline([
#         ("poly_features", polybig_features),
#         ("std_scaler", std_scaler),
#         ("lin_reg", lin_reg),
#     ])
#     polynomial_reg.fit(X, y)
#     y_newbig = polynomial_reg.predict(X_new)
#     plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

# plt.plot(X, y, "b.")
# plt.axis([-3, 3, 0, 10])
# plt.plot(X_new, y_new, "r-")

# def plot_learning_curves(model, X, y):
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
#     train_error, val_error = [], []
#     for m in range(1, len(X_train)):
#         model.fit(X_train[:m], y_train[:m])
#         y_train_predict = model.predict(X_train[:m])
#         y_val_predict = model.predict(X_val)
#         train_error.append(mean_squared_error(y_train[:m], y_train_predict))
#         val_error.append(mean_squared_error(y_val, y_val_predict))
#     plt.plot(np.sqrt(train_error), "r-+")
#     plt.plot(np.sqrt(val_error), "b-")
#
#
# polynomial_reg = Pipeline([
#     ("poly_feature", PolynomialFeatures(degree=10, include_bias=False)),
#     ("lin_reg", LinearRegression())
# ])
# lin_reg = LinearRegression()
# plot_learning_curves(polynomial_reg, X, y)
# plt.axis([0, 80, 0, 3])
# plt.show()

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg_pre = ridge_reg.predict([[1.5]])
print(ridge_reg_pre)

sgd_reg = SGDRegressor(max_iter=5, penalty="l2")
sgd_reg.fit(X, y)
sgd_reg_pre = sgd_reg.predict([[1.5]])
print(sgd_reg_pre)

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg_pre = lasso_reg.predict([[1.5]])
print(lasso_reg_pre)

elastic_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_reg.fit(X, y)
print(elastic_reg.intercept_, elastic_reg.coef_)
print(elastic_reg.predict([[1.5]]))

iris = datasets.load_iris()

# X = iris["data"][:, 3:]
# y = (iris["target"] == 2).astype(np.int)
# log_reg = LogisticRegression()
# log_reg.fit(X, y)
#
# X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# y_proba = log_reg.predict_proba(X_new)
# plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
# plt.legend(loc="center left", fontsize=14)
# plt.show()
#
# print(log_reg.predict([[1.7], [1.5]]))

X = iris["data"][:, (2, 3)]
y = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, y)
print(softmax_reg.predict([[5, 2]]))
print(softmax_reg.predict_proba([[5, 2]]))

x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),
    np.linspace(0, 3.5, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]
# print(np.linspace(0, 8, 500))
# print(np.linspace(0, 8, 500).reshape(-1, 1))
print(x0.ravel())
print(x1.ravel())
print(X_new)

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Length", fontsize=14)
plt.ylabel("Width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()
