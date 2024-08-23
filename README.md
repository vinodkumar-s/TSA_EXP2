### DEVELOPED BY: VINOD KUMAR S
### REGISTER NO: 212222240116
### DATE:

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

## AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

## ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program

## PROGRAM:
```python
# LINEAR TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('coffeesales.csv',nrows=50)
data['date'] = pd.to_datetime(data['date'])
daily_average = data.groupby('date')['money'].mean().reset_index()

# Linear trend estimation
x = np.arange(len(daily_average))
y = daily_average['money']
linear_coeffs = np.polyfit(x, y, 1)
linear_trend = np.polyval(linear_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(daily_average['date'], daily_average['money'], label='Original Data', marker='o')
plt.plot(daily_average['date'], linear_trend, label='Linear Trend', color='red')
plt.title('Linear Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Average Money')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# POLYNOMIAL TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('coffeesales.csv'nrows=50)
data['date'] = pd.to_datetime(data['date'])
daily_average = data.groupby('date')['money'].mean().reset_index()

# Polynomial trend estimation (degree 2)
x = np.arange(len(daily_average))
y = daily_average['money']
poly_coeffs = np.polyfit(x, y, 2)
poly_trend = np.polyval(poly_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(daily_average['date'], daily_average['money'], label='Original Data', marker='o')
plt.plot(daily_average['date'], poly_trend, label='Polynomial Trend (Degree 2)', color='green')
plt.title('Polynomial Trend Estimation')
plt.xlabel('Date')
plt.ylabel('Average Money')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


```

## OUTPUT
A - LINEAR TREND ESTIMATION

![Screenshot 2024-08-23 215256](https://github.com/user-attachments/assets/f452c45a-e58e-46fd-98cb-6c0e311a5d7f)


B- POLYNOMIAL TREND ESTIMATION


![Screenshot 2024-08-23 215311](https://github.com/user-attachments/assets/18fbdfe7-2e2b-45ee-9005-03cd3b64ecf2)

## RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
