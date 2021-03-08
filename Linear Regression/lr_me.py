import numpy as np


class LinearRegression:

    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr  # learning rate
        self.iters = iters  # iterations
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.bias = 0
        self.weight = np.random.rand(n_features)

        # gradient descent
        for i in range(self.iters):
            y_predict = np.dot(X, self.weight) + self.bias

            self.weight -= self.lr * (1 / m_samples) * np.dot(X.T, (y_predict - y))
            self.bias -= self.lr * (1 / m_samples) * np.sum(y_predict - y)

    def predict(self, X):
        return np.dot(X, self.weight) + self.bias



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    plt.style.use('seaborn')

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression(lr=1,iters=100)
    regressor.fit(X_train, y_train)
    pd_line = regressor.predict(X_train)
    #print(regressor.bias)

    y_predict_test = regressor.predict(X_test)

    def mse(y_predict, y_true):
        return np.sum((y_predict-y_true)**2)

    mse_value = mse(y_predict_test,y_test)

    print(mse_value)

    plt.scatter(X_train,y_train,color='green',label='Train')
    plt.scatter(X_test,y_test,color='blue',label='Test')
    plt.plot(X_train,pd_line,color='red',label='Prediction Line')

    plt.title('Linear Regression 1 Feature/100 Samples')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()
