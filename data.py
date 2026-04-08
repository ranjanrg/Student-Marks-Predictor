import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


data = {
  'hours' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
  'marks' : [35, 40, 45, 50, 70, 75, 80, 85, 90, 100 ]
}

df = pd.DataFrame(data)

# Separation of data
x = df[['hours']]
y = df['marks']

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# creating the model.
model = LinearRegression()

# training the model
model.fit(x_train, y_train)

# make prediction
prediction = model.predict(x_test)
y_pred_line = model.predict(x)

print("Predictions:", prediction)
print("Actual:", y_test.values)


# Mean absolute error
error = mean_absolute_error(y_test, prediction)
print("Mean absolute error: ", error)

# visualize the preiction
plt.scatter(x, y)
plt.plot(x, y_pred_line)
plt.xlabel("Hours Studied")
plt.title = ("Hours vs Marks")
plt.ylabel = ("Marks Scored")
plt.show()

