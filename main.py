# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# datase importing 
dataset = pd.read_csv("Regression/multiple_regresion/50_Startups.csv")
# below line means all row and all columns except the last column i.e prediction(output)
x = dataset.iloc[:, :-1].values
# below line means all row of the last column i.e the predicted values 
y = dataset.iloc[:, -1].values

# encodeing the dataset
# in this data set state is encoded the index of state ia 3 here 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(handle_unknown='ignore'),[3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))


# importing the libraries for data splitting for training and testing purpose
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# training the multi linear regression model on the training set

from sklearn.linear_model import LinearRegression
# below line build the multiple regression model 
regressor = LinearRegression()
# this line trian on training set
regressor.fit(x_train,y_train)


# predicting the test set results

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print("pred_profit->original_profit")

# below line give the prediction for the test data 
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# for new data

new_startup = np.array([[160000, 130000, 300000, 'California']], dtype=object)

new_startup_encoded = ct.transform(new_startup)

predicted_profit = regressor.predict(new_startup_encoded)

print("Predicted Profit:", predicted_profit[0])
