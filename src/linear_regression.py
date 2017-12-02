import numpy as np

X = np.array([[6], [8], [10], [14], [18]]).reshape(-1, 1)
x_bar = X.mean()
print(x_bar)

X - x_bar
# calculating sample variance with Bessel's correction
variance = ((X - x_bar) ** 2).sum() / (X.shape[0] - 1)
print(variance)
print(np.var(X, ddof=1))

y = np.array([7, 9, 13, 17.5, 18])
y_bar = y.mean()
covariance = np.multiply((X - x_bar).transpose(), (y - y_bar)).sum() / (X.shape[0] - 1)
print(covariance)
print(np.cov(X.transpose(), y)[0][1])

from sklearn.linear_model import LinearRegression

X_train = np.array([6, 8, 10, 14, 18]).reshape(-1, 1)
y_train = [7, 9, 13, 17.5, 18]

X_test = np.array([8, 9, 11, 16, 12]).reshape(-1, 1)
y_test = [11, 8.5, 15, 18, 11]

model = LinearRegression()
model.fit(X_train, y_train)
r_squared = model.score(X_test, y_test)
print(r_squared)
