import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

class LinearRegr:
    def fit(self, X, Y):
        # input:
        #  X = np.array, shape = (n, m)
        #  Y = np.array, shape = (n)
        # Finds theta that minimalizes cost function L 
        n, m = X.shape
        self.theta = np.zeros((m+1))
        X = np.concatenate((np.ones(n)[:, np.newaxis], X), axis=1)
        print(f"\nFIT \n ------------------------")
        print(f"X: \n{X}")
        X_T = X.T
        self.theta = np.linalg.inv(X_T @ X) @ X_T @ Y  
        print(f"THETA: {self.theta}")

        return self

    def predict(self, X):
        # input
        #  X = np.array, shape = (k, m)
        # returns 
        #  Y = vector:(f(X_1), ..., f(X_k))
        k, m = X.shape
        X = np.concatenate((np.ones(k)[:, np.newaxis], X), axis=1)
        Y_hat = X @ self.theta
        return Y_hat


def test_RegressionInOneDim():
    X = np.array([1,3,2,5]).reshape((4,1))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,2,10]).reshape((3,1))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    print("\nTest 1 Dim \n ------------------")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    assert list(actual) == pytest.approx(list(expected))

def test_RegressionInThreeDim():
    X = np.array([1,2,3,5,4,5,4,3,3,3,2,5]).reshape((4,3))
    Y = np.array([2,5, 3, 8])
    a = np.array([1,0,0, 0,1,0, 0,0,1, 2,5,7, -2,0,3]).reshape((5,3))
    expected = LinearRegression().fit(X, Y).predict(a)
    actual = LinearRegr().fit(X, Y).predict(a)
    print("\nTest 3 Dim \n --------------------")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    assert list(actual) == pytest.approx(list(expected))

test_RegressionInOneDim()
test_RegressionInThreeDim()