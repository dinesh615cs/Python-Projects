import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

data=pd.read_csv("hoursmarks.csv")

x=np.array(data["hours"]).reshape(-1,1)
y=np.array(data["marks"])

input_data=x
output_data=y

model=LinearRegression()

model.fit(input_data,output_data)

print(model.predict([[6]]))