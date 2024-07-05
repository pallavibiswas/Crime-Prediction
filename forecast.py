import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('transformed_data/feature_engineered_oncampus_result.csv')

# Identify the relevant columns (e.g., incidents per year)
# Ensure that the data has a 'YEAR' column and columns for each type of incident
df['YEAR'] = pd.to_datetime(df['YEAR'], format='%Y')

# Aggregate the data by year if not already done
yearly_data = df.groupby(df['YEAR'].dt.year).sum().reset_index()

# Define the columns to forecast (all incident types)
incident_columns = [col for col in df.columns if col not in ['YEAR', 'FILTER20', 'FILTER21', 'FILTER22']]

# Forecast function
def forecast_incidents(data, columns, forecast_years=2):
    forecasts = {}
    for column in columns:
        # Fit the model
        model = ExponentialSmoothing(data[column], seasonal='add', seasonal_periods=1).fit()
        
        # Forecast the future values
        forecast = model.forecast(forecast_years)
        
        # Store the forecasted values
        forecasts[column] = forecast
    
    return forecasts

# Perform the forecasting
forecasts = forecast_incidents(yearly_data, incident_columns, forecast_years=2)

# Convert forecasts to DataFrame
forecast_years = np.arange(yearly_data['YEAR'].max() + 1, yearly_data['YEAR'].max() + 3)
forecast_df = pd.DataFrame(forecasts, index=forecast_years)

# Plot the results
plt.figure(figsize=(12, 8))
for column in incident_columns:
    plt.plot(yearly_data['YEAR'], yearly_data[column], label=f'Historical {column}')
    plt.plot(forecast_df.index, forecast_df[column], '--', label=f'Forecasted {column}')

plt.xlabel('Year')
plt.ylabel('Number of Incidents')
plt.title('Historical and Forecasted Incidents')
plt.legend()
plt.grid(True)
plt.show()

# Save the forecasted results
forecast_df.to_csv('/path/to/save/forecasted_incidents.csv', index=True)