import numpy as np
import pandas as pd
import seaborn as sns
    
csv=pd.read_csv("CarPrice_Assignment.csv")
csv.drop('CarName',axis=1, inplace=True)
csv = pd.get_dummies(csv, columns=['fueltype', 'aspiration', 'doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem'])
from sklearn.preprocessing import StandardScaler
y = csv['price']
scaler = StandardScaler()
x = scaler.fit_transform(csv.drop(columns=["price"]))
x = pd.DataFrame(data=x, columns=csv.drop(columns=["price"]).columns)
k=x.corr()
z = [(str(k.columns[i]), str(k.columns[j])) for i in range(len(k.columns)) for j in range(i+1, len(k.columns)) if abs(k.corr().iloc[i, j]) > 0.5]
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = x
vif = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])], index=vif_data.columns)
def remover(csv):
    vif = pd.Series([variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])], index=vif_data.columns)
    if vif.max() == float('inf') or vif.max() > 4.6:
        column_to_drop = vif[vif == vif.max()].index[0]
        csv=csv.drop(columns=[column_to_drop])
    else:
        pass
    return csv
for i in range(50):
    vif_data=remover(vif_data)

X=vif_data
y=csv['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
logmodel = LinearRegression()

logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predictions = logmodel.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)  # RMSE
r_squared = r2_score(y_test, predictions)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

predictions = logmodel.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)  # RMSE
r_squared = r2_score(y_test, predictions)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r_squared)