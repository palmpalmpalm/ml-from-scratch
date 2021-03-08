import numpy as np

class svm:

    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param - lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y<= 0,-1,1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):
            for index, x_i in enumerate(X):
                condition = y_[index] * (np.dot(x_i,self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i,y[index]))
                    self.b -= self.lr * y_[index]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np,sign(linear_output)

