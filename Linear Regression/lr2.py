import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from linear_regression import LinearRegression
from lr_me import LinearRegression

rng = np.random

x = rng.rand(50) * 10
y = 2 * x + rng.random(50)
x_new = x.reshape(-1, 1)

# lr model
model = LinearRegression()

# train
model.fit(x_new, y)

# test model
xpredict = np.linspace(-1, 11)
xpredict_new = xpredict.reshape(-1, 1)

ypredict = model.predict(xpredict_new)

# analysis model
plt.scatter(x,y)
plt.plot(xpredict,ypredict,color='red')

plt.tight_layout()

plt.show()

# plt.scatter(x,y)
# plt.tight_layout()
# plt.xlabel('x')
# plt.ylabel('y')
#
# plt.show()
