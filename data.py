import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
  'hours' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  'marks' : [35, 40, 45, 50, 70, 75, 80, 85, 90, 100 ]
}

df = pd.DataFrame(data)

# Separation of data
x = df[['hours']]
y = df['marks']


# creating the model.
model = LinearRegression()

# training the model
model.fit(x, y)

# make prediction
prediction = model.predict(pd.DataFrame({'hours': [5]}))
print(prediction)
