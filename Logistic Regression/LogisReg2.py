import numpy as np

# def sigmoid(x):
#     #t = np.array(x, dtype=np.float128)
#     return 1 / (1 + np.exp(-x))


class LogisticRegression:

    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.weight = np.random.rand(n_features)
        self.bias = 0

        for i in range(self.iters):
            linear_model = np.dot(X, self.weight) + self.bias
            y_predict = self.sigmoid(linear_model)

            self.weight -= self.lr * (1 / m_samples) * np.dot(X.T, (y_predict - y))
            self.bias -= self.lr * (1 / m_samples) * np.sum(y_predict - y)

    def predict(self, X):
        linear_model = np.dot(X, self.weight) + self.bias
        y_predict = self.sigmoid(linear_model)
        y_predict_cls = []
        for pd in y_predict:
            if pd >= 0.5:
                y_predict_cls.append(1)
            else:
                y_predict_cls.append(0)

        return y_predict_cls

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


    def acc(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc


    regressor = LogisticRegression(lr=0.01, iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print(acc(y_test, predictions))
