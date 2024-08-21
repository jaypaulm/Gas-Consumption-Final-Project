#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('BP_dataset.csv')

BP_dataset = pd.DataFrame(data)

#Preparing seasonal values
seasonal_values = BP_dataset.copy()
seasonal_values['season'] = seasonal_values['month'].apply(lambda x: 'Winter' if x in ['12', '01', '02'] else ('Spring' if x in ['03', '04', '05'] else ('Summer' if x in ['06', '07', '08'] else 'Fall')))
seasonal_values_ex2024 = seasonal_values[:14]  # For simplicity, let's use the first 14 rows for demonstration

#Aggregat values for each year and season
seasonal_values_ex2024_agg = seasonal_values_ex2024.groupby(['year', 'season'])['value'].sum().reset_index()

# Pivoting the DataFrame
seasonal_values_ex2024_pivot = seasonal_values_ex2024_agg.pivot(index='year', columns='season', values='value')



# ### Variables
# 
# #### BP_dataset: A DataFrame created as a copy of data for operations.
# #### Seasonal_values: A DataFrame derived from BP_dataset where a new column for seasons is added based on the month.
# #### x_labels: List of strings combining year and season for custom x-axis labels in a plot. 
# #### future_years: An array containing additional years for which predictions are made.   
# 
#      

# #### Annual consumption of natural gas in the US in millions of cubic feet (MMCF)

# In[2]:


df = pd.read_csv('BP_dataset.csv')
df.head()
annual_value = df.groupby('year')['value'].sum()
annual_value


# In[32]:


annual_value_without_2024 = annual_value.loc[:2023] 
fig, ax = plt.subplots(figsize = (8,6))
ax.plot(annual_value_without_2024, marker = 'o')
plt.xlabel('Year')
plt.ylabel('Consumption of natural gas in millions of cubic feet (MMCF)')
plt.title('Annual consumption of natural gas in the US')
plt.grid(True)
plt.show()


# #### Seasonal consumption of natural gas in the US

# In[4]:


# Assuming df is DataFrame
# Defining a function to map months to seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

#Creating a new column for seasons
df['season'] = df['month'].apply(get_season)

#Grouping by year and season, then sum the values
seasonal_values = df.groupby(['year', 'season'])['value'].sum().reset_index()

print(seasonal_values)


# In[5]:


seasonal_values_ex2024 = seasonal_values[:40]
seasonal_values_ex2024_pivot = seasonal_values_ex2024.pivot(index='year', columns='season', values='value')

# Plot the values against the year for each season
seasonal_values_ex2024_pivot.plot(marker='o', linestyle='-')

# Add labels and title
plt.title('Seasonal consumption of natural gas by season in the US')
plt.xlabel('Year')
plt.ylabel('Seasonal consumption of natural gas in MMCF')
plt.grid(True)
plt.legend(title='Season')
plt.tight_layout()
plt.show()


# In[6]:


df['process-name'].unique()


# In[7]:


consumption_by_use = df.groupby('process-name')['value'].sum()
consumption_by_use


# In[8]:


fig, ax = plt.subplots(figsize = (8,6))
ax.plot(consumption_by_use, marker = 'o')
plt.xticks(rotation = 45, ha = 'right')
plt.xlabel('Consumption using case')
plt.ylabel('Consumption value')
plt.title('Natural gas consumption in the US by consumption use')
plt.grid(True)
plt.show()


# In[9]:


seasonal_cases = df.groupby(['year', 'season', 'process-name'])['value'].sum().reset_index()
seasonal_cases


# In[10]:


seasonal_cases_pivot = seasonal_cases.pivot(index=['year', 'season'], columns='process-name', values='value')
seasonal_cases_pivot.index.shape


# In[11]:


#Assuming seasonal_cases_pivot is your DataFrame
#Combining 'year' and 'season' into a single string for x-axis labels
x_labels = [f"{year}-{season}" for year, season in seasonal_cases_pivot.index]

#Plot the values against the year for each season
plt.figure(figsize=(14, 8))  # Set the figsize
seasonal_cases_pivot.plot(marker='o', linestyle='-')

#Adding labels and title
plt.title('Seasonal consumption of natural gas by process in the US')
plt.xlabel('Year and season')
plt.ylabel('Seasonal consumption of natural gas in MMCF')

#Show all x-ticks labels
plt.xticks(range(len(x_labels)), x_labels, rotation=90, ha='center')

#Ensure all x-tick labels are shown without overlap
plt.tight_layout()

#Minimize the legend size
plt.legend(title='Consumption using case', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.grid(True)
plt.show()


# #### Annual consumption by process

# In[12]:


annual_consumption = df.groupby(['year', 'process-name'])['value'].sum().reset_index()
pivot_df = annual_consumption.pivot(index='year', columns='process-name', values='value')
pivot_df


# In[ ]:





# ### Forecasted MSE for the year 2024

# In[18]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
data = pd.read_csv('BP_dataset.csv')

# Convert 'year' to datetime and extract year if not already in year format
data['year'] = pd.to_datetime(data['year'], format='%Y').dt.year

# Exclude data from 2024
data = data[data['year'] != 2024]

# Aggregate data by year, summing the gas consumption
annual_data = data.groupby('year')['value'].sum().reset_index()

# Reshape the data for modeling
X = annual_data['year'].values.reshape(-1,1)
y = annual_data['value'].values

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

#Predicting for 2024
year_2024 = np.array([[2024]])
predicted_consumption_2024 = model.predict(year_2024)
print(f"Forecasted Gas Consumption for 2024: {predicted_consumption_2024[0]} MMCF")

from sklearn.metrics import mean_squared_error, r2_score

#Predict on the known data to evaluate the model
y_pred = model.predict(X)

#Calculating metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


# ### Forecasted model 

# In[19]:


import matplotlib.pyplot as plt

#Forecasting for additional years if needed
future_years = np.array([[year] for year in range(2025, 2030)])  # Forecasting till 2029 for a broader view
future_predictions = model.predict(future_years)

#Combining all years and predictions for plotting
all_years = np.concatenate((X, year_2024, future_years))
all_predictions = np.concatenate((y_pred, predicted_consumption_2024, future_predictions))

#Plot the actual data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')  # Actual data points

#Ploting the regression line including forecasts
plt.plot(all_years, all_predictions, color='red', linestyle='-', linewidth=2, label='Regression & Forecast')

#Highlight the forecast point
plt.scatter(year_2024, predicted_consumption_2024, color='green', s=100, label='2024 Forecast')

#Enhancing the plot
plt.title('Historical Gas Consumption and Forecast')
plt.xlabel('Year')
plt.ylabel('Gas Consumption (MMCF)')
plt.grid(True)
plt.legend()

#Show plot
plt.show()


# ### Regression model until 2023

# In[15]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('BP_dataset.csv')

#Assuming the dataset includes a 'year' column and a 'value' column for gas consumption
data['year'] = pd.to_datetime(data['year'].astype(str), format='%Y').dt.year  # Ensuring 'year' is an integer

#Excluding the year 2024
data = data[data['year'] != 2024]

print(data.head())


# In[16]:


#Aggregate gas consumption by year
annual_data = data.groupby('year')['value'].sum().reset_index()

#Preparing independent and dependent variables
X = annual_data['year'].values.reshape(-1, 1)  # Reshape for sklearn
y = annual_data['value'].values

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Predicting on the test set
y_pred = model.predict(X_test)

#Calculating metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


# In[17]:


# Plotting regression line and actual data points
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Annual Gas Consumption and Regression Line')
plt.xlabel('Year')
plt.ylabel('Total Gas Consumption (MMCF)')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Example DataFrame creation
data = {
    'year': np.random.randint(2000, 2023, size=100),
    'value': np.random.rand(100) * 1000  # Random numeric data
}
df = pd.DataFrame(data)

# Convert year to a numeric type explicitly
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df.dropna(inplace=True)

# Add a constant column for VIF calculation
df = add_constant(df)

# Calculate VIF
vif_data = pd.DataFrame({
    'Feature': df.columns,
    'VIF': [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
})

print(vif_data)


# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Prepare the data
X = filtered_data['year'].values.reshape(-1, 1)
y = filtered_data['value'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and calculate metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Adjusted Mean Squared Error: {mse}')
print(f'Adjusted R² Score: {r2}')

# Visualize the regression line with the adjusted dataset
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.xlabel('Year')
plt.ylabel('Gas Consumption (MMCF) [Filtered]')
plt.title('Regression Analysis After Removing Outliers')
plt.grid(True)
plt.show()


# In[31]:


from sklearn.preprocessing import PolynomialFeatures

# Generating polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)

# Fit a linear regression model on the polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

# Use the model to make predictions on the test set
X_test_poly = poly.transform(X_test)
y_pred_poly = poly_model.predict(X_test_poly)

# Calculate the new metrics
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f'Polynomial Regression Mean Squared Error: {mse_poly}')
print(f'Polynomial Regression R² Score: {r2_poly}')

# Visualize the regression fit
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.scatter(X_test, y_pred_poly, color='red', label='Predicted Data', alpha=0.5)
plt.legend()
plt.title('Polynomial Regression Fit')
plt.xlabel('Year')
plt.ylabel('Gas Consumption (MMCF)')
plt.show()


# In[ ]:




