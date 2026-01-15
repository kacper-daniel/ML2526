import numpy as np
import pytest
from sklearn.linear_model import Ridge

class RidgeRegr:
    def __init__(self, alpha = 0.0, lr = 0.01):
        self.alpha = alpha
        self.lr = lr

    def fit(self, X, Y):
        # input:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Finds theta (approx) that minimizes cost function L
        n, m = X.shape
        self.theta = np.zeros((m+1))
        X = np.concatenate((np.ones(n)[:, np.newaxis], X), axis=1)
        print(f"\nFIT \n ------------------------")
        print(f"X: \n{X}")

        self.Y_hat = X @ self.theta
        loss_prev = -np.inf
        loss_value = self.loss(Y, self.Y_hat)

        iterations = 1
        while abs(loss_value - loss_prev) >= 0.0000000001:
            theta_grad = self.theta.copy()
            theta_grad[0] = 0

            grad = -(X.T @ (Y - self.Y_hat)) + self.alpha * theta_grad
            self.theta -= self.lr * grad

            self.Y_hat = X @ self.theta
            loss_prev = loss_value
            loss_value = self.loss(Y, self.Y_hat)

            iterations += 1

        print(f"THETA: {self.theta}")
        print(f'iterations: {iterations}')

        return self
    
    def loss(self, Y, Y_hat):
        return (1/2) * np.sum((Y-Y_hat)**2) + (1/2) * self.alpha * np.sum((self.theta)**2)

    def predict(self, X):
        # input
        #  X = np.array, shape = (k, m)
        # returns
        #  Y = vector: (f(X_1), ..., f(X_k))
        k, m = X.shape
        X = np.concatenate((np.ones(k)[:, np.newaxis], X), axis=1)
        Y_hat = X @ self.theta
        return Y_hat


def test_RidgeRegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,2,10]).reshape((3,1))
    alpha = 0.3
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    print("\nTest 1 Dim \n ------------------")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    assert list(actual) == pytest.approx(list(expected), rel=1e-5)

def test_RidgeRegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    X_test = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    alpha = 0.4
    expected = Ridge(alpha).fit(X, Y).predict(X_test)
    actual = RidgeRegr(alpha).fit(X, Y).predict(X_test)
    print("\nTest 3 Dim \n --------------------")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    assert list(actual) == pytest.approx(list(expected), rel=1e-3)
    
test_RidgeRegressionInOneDim()
test_RidgeRegressionInThreeDim()