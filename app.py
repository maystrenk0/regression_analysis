import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('COVID-19-in-Ukraine-from-April.csv')
data = data.dropna()

x = data['n_confirmed']
y = data['n_deaths']

train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x, y, test_size=0.33, random_state=42)

linear_regressor = LinearRegression()
linear_regressor.fit(X=train_set_x.to_numpy().reshape(-1, 1), y=train_set_y.to_numpy().reshape(-1, 1))
Y_pred = linear_regressor.predict(test_set_x.to_numpy().reshape(-1, 1))
print('Coefficients: \n', linear_regressor.coef_)
print('Mean squared error: %.2f' % mean_squared_error(test_set_y.to_numpy().reshape(-1, 1), Y_pred))
print('Coefficient of determination: %.2f' % r2_score(test_set_y.to_numpy().reshape(-1, 1), Y_pred))

plt.title("train_set")
plt.scatter(train_set_x, train_set_y)
x = np.array([min(train_set_x), max(train_set_x)])
y = linear_regressor.predict(x.reshape(-1, 1))
plt.plot(x, y, color='orange')
plt.xlabel('n_confirmed')
plt.ylabel('n_deaths')
plt.show()

plt.title("test_set")
plt.scatter(test_set_x, test_set_y)
x = np.array([min(test_set_x), max(test_set_x)])
y = linear_regressor.predict(x.reshape(-1, 1))
plt.plot(x, y, color='orange')
plt.xlabel('n_confirmed')
plt.ylabel('n_deaths')
plt.show()

sns.residplot(Y_pred.reshape(-1), test_set_y, lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
plt.xlabel("Fitted values")
plt.title('Residual plot')
plt.show()

residuals = test_set_y - Y_pred.reshape(-1)
plt.figure(figsize=(7, 7))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot")
plt.show()

model_norm_residuals_abs_sqrt = np.sqrt(np.abs(residuals))
plt.figure(figsize=(7, 7))
sns.regplot(Y_pred.reshape(-1), model_norm_residuals_abs_sqrt,
            scatter=True,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plt.ylabel("Standarized residuals")
plt.xlabel("Fitted value")
plt.show()
