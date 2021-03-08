import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn')

data = pd.read_csv('Weather.csv')
x = data['MinTemp'].values.reshape(-1,1)
y = data['MaxTemp'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#train
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#test
y_predict = regressor.predict(x_test)


# compare
data2 = pd.DataFrame({'Actually':y_test.flatten(),'Predicted':y_predict.flatten()})

dataplot = data2.head(20)
dataplot.plot(kind='bar',figsize=(16,10))
plt.show()


# plt.scatter(x_test,y_test,color='red',marker='.')
# plt.plot(x_test,y_predict,color='blue')
#
# plt.tight_layout()
# plt.show()

# plt.scatter(data['MinTemp'],data['MaxTemp'],marker='.',color='green')
# plt.title('Min & Max Temp')
# plt.xlabel('Mintemp')
# plt.ylabel('Maxtemp')
#
# plt.tight_layout()
# plt.show()