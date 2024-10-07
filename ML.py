import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

d1=pd.read_csv("Car_sales_details_p3.csv")
d2=pd.read_csv("sales_Status_p3.csv")
d3=pd.read_csv("State_region_mapping_p3.csv")

d1 = pd.merge(d1, d2, on='Sales_ID', how='inner')
merged_df = pd.merge(d1, d3, on='State or Province', how='inner')

merged_df.rename(columns={'State or Province':'State_or_Province'},inplace=True)
merged_df['mileage']=merged_df['mileage'].str.replace('km/kg','')
merged_df['engine']=merged_df['engine'].str.replace('CC','')
merged_df['max_power']=merged_df['max_power'].str.replace('bhp','')

#QUANTILE=25%
max_power_q1= merged_df['max_power'].quantile(0.25)
# print(f"25% quantile of max_power: {max_power_q1} ")

engine_q1= merged_df['engine'].quantile(0.25)
# print(f"25% quantile of engine: {engine_q1} ")


mileage_q1= merged_df['mileage'].quantile(0.25)
# print(f"25% quantile of mileage: {mileage_q1} ")

#QUANTILE=50%
max_power_q2= merged_df['max_power'].quantile(0.5)
# print(f"50% quantile of max_power: {max_power_q2} ")

engine_q2= merged_df['engine'].quantile(0.5)
# print(f"50% quantile of engine: {engine_q2} ")


mileage_q2= merged_df['mileage'].quantile(0.5)
# print(f"50% quantile of mileage: {mileage_q2} ")

#QUANTILE=75%
max_power_q3= merged_df['max_power'].quantile(0.75)
# print(f"75% quantile of max_power: {max_power_q3} ")

engine_q3= merged_df['engine'].quantile(0.75)
# print(f"75% quantile of engine: {engine_q3} ")


mileage_q3= merged_df['mileage'].quantile(0.75)
# print(f"75% quantile of mileage: {mileage_q3} ")
max_power_iqr=max_power_q3-max_power_q1
max_power_uw=max_power_q3+1.5*max_power_iqr
max_power_lw=max_power_q1-1.5*max_power_iqr

# print(f"max_power_iqr={max_power_iqr}")
# print(f"max_power_uw = {max_power_uw}")
# print(f"max_power_lw={max_power_lw}")

engine_iqr=engine_q3-engine_q1
engine_uw=engine_q3+1.5*engine_iqr
engine_lw=engine_q1-1.5*engine_iqr

# print(f"engine_iqr={engine_iqr}")
# print(f"engine_uw = {engine_uw}")
# print(f"engine_lw={engine_lw}")

mileage_iqr=mileage_q3-mileage_q1
mileage_uw=mileage_q3+1.5*mileage_iqr
mileage_lw=mileage_q1-1.5*mileage_iqr

# print(f"mileage_iqr={mileage_iqr}")
# print(f"mileage_uw = {mileage_uw}")
# print(f"mileage_lw={mileage_lw}")

label_encoder = LabelEncoder()
merged_df['transmission'] = label_encoder.fit_transform(merged_df['transmission'])
# print(merged_df['transmission'].unique())

merged_df['fuel'] = label_encoder.fit_transform(merged_df['fuel'])
# print(merged_df['fuel'].unique())

merged_df['seller_type'] = label_encoder.fit_transform(merged_df['seller_type'])
# print(merged_df['seller_type'].unique())

def remove_outliers_iqr(df, column):
     Q1 = df[column].quantile(0.25)     
     Q3 = df[column].quantile(0.75)     
     IQR = Q3 - Q1    
     lower_bound = Q1 - 1.5 * IQR 
     upper_bound = Q3 + 1.5 * IQR
     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

merged_df['sold'] = label_encoder.fit_transform(merged_df['sold'])
# print(merged_df['sold'].unique())

merged_df['Region'] = label_encoder.fit_transform(merged_df['Region'])
# print(merged_df['Region'].unique())

merged_df['owner'] = label_encoder.fit_transform(merged_df['owner'])
# print(merged_df['owner'].unique())

l=merged_df[['km_driven','mileage', 'engine', 'max_power', 'seats','sold','selling_price']]

df=remove_outliers_iqr(l, 'mileage')


import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# Load your dataset
# df = pd.read_csv('your_dataset.csv')  # Uncomment and modify this line to load your dataset

# Assuming df is your DataFrame
X = merged_df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'mileage', 'engine', 'max_power', 'seats']]
y = merged_df['selling_price']

# Encode categorical variables
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X[['fuel', 'seller_type', 'transmission']]).toarray()

# Combine encoded features with numerical features
X_numerical = X[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']].values
X_combined = np.hstack((X_numerical, X_encoded))

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X_combined)

# Log transformation
X_log = np.log(X_combined + 1)  # Adding 1 to avoid log(0)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined)

# Combine all features
X_final = np.hstack((X_combined, X_poly, X_log, X_scaled, X_pca))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

# Display metrics
# print("MAE:", mae)
# print("MSE:", mse)
# print("RMSE:", rmse)
# print("R²:", r2)
# print("MAPE:", mape)

# Create a DataFrame to compare actual and predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
# print(comparison_df.head())
