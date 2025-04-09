
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
# from tabulate import tabulate
def load_and_prepare_data():
    # Load and combine data
    data1 = pd.read_csv("data/Ohio GDP 2021 new.csv").apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    data2 = pd.read_csv("data/ohio GDP 2022 New.csv").apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # data3 = pd.read_csv("data/Ohio GDP 2023new.csv").apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # data4 = pd.read_csv("data/Ohio GDP 2024N.csv").apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    df = pd.concat([data1, data2], ignore_index=True)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Unpivot the data to long format
    df_long = df.melt(id_vars=['Description'],
                      value_vars=[
                          '2021:Q1', '2021:Q2', '2021:Q3', '2021:Q4',
                          '2022:Q1', '2022:Q2', '2022:Q3', '2022:Q4',],
                        #   '2023:Q1', '2023:Q2', '2023:Q3', '2023:Q4',
                        #   '2024:Q1', '2024:Q2', '2024:Q3', '2024:Q4'],
                      var_name='Quarter',
                      value_name='Value')

    # Drop rows where Value is missing
    df_long = df_long.dropna(subset=['Value'])

    # Split 'Quarter' into 'Year' and 'Quarter_Label'
    df_long[['Year', 'Quarter_Label']] = df_long['Quarter'].str.split(':', expand=True)
    df_long['Year'] = df_long['Year'].astype(int)

    #'Value' to 'GDP'
    df_long = df_long.rename(columns={'Value': 'GDP'})

    # Drop 'Quarter' and 'Date' columns since they're not needed
    df_long = df_long.drop(columns=['Quarter'])

    # Sort and return final cleaned data
    return df_long.sort_values(['Description', 'Year', 'Quarter_Label'])

# Call the function and store the result
df_long = load_and_prepare_data()
# Pivot so each Description is a row, and each Year+Quarter is a column
pivoted_df = df_long.pivot(index='Description', columns=['Year', 'Quarter_Label'], values='GDP')

# Optional: sort columns for cleaner view
pivoted_df = pivoted_df.sort_index(axis=1, level=[0,1])  # Sort by Year then Quarter

# # Print top few rows
# print(tabulate(pivoted_df.head(5), headers='keys', tablefmt='grid'))




# %%
# ==== Filter for one industry ====

industry_df = df_long[df_long['Description'] == 'All industry total'].copy()
industry_df = industry_df.sort_values(['Year', 'Quarter_Label'])

# Optional: convert Quarter_Label to a number for modeling
quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
industry_df['Quarter_Num'] = industry_df['Quarter_Label'].map(quarter_map)

# Create a time index and target
industry_df['Time_Index'] = np.arange(len(industry_df))
industry_df['target'] = industry_df['GDP'].shift(-1)
industry_df = industry_df.dropna()

# ==== Supervised learning data ====
features = ['Time_Index', 'Quarter_Num']  # Numeric features only
X = industry_df[features].values
y = industry_df['target'].values

# ==== Train/test split ====
def train_test_split(X, y, test_size=0.2):
    n = int(len(X) * (1 - test_size))
    return X[:n], X[n:], y[:n], y[n:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ==== Train XGBoost model ====
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(X_train, y_train)

# ==== Make prediction ====
val = X_test[0].reshape(1, -1)
pred = model.predict(val)
print(f"Predicted GDP: {pred[0]}")

# predict train on train set and predic one sample at a time

def xgb_predict(train, val):
    train = np.array(train)
    X, y = train[:, :-1], train[:, -1]
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(X, y)

    val = np.array(val).reshape(1, -1)
    pred = model.predict(val)
    return pred[0]

train = np.column_stack((X_train, y_train))
test_sample = X_test[0]
# print(xgb_predict(train, test_sample))


def forecast_industry(industry_name='All industry total', target_year=None, target_quarter='Q4'):
    df_long = load_and_prepare_data()
    industry_df = df_long[df_long['Description'] == industry_name].copy()
    industry_df = industry_df.sort_values(['Year', 'Quarter_Label'])

    # Map quarters to numbers (Q1=1, Q2=2, etc.)
    quarter_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    industry_df['Quarter_Num'] = industry_df['Quarter_Label'].map(quarter_map)
    
    # Create a time index (0, 1, 2, ...)
    industry_df['Time_Index'] = np.arange(len(industry_df))
    
    # Target is the next quarter's GDP
    industry_df['target'] = industry_df['GDP'].shift(-1)
    industry_df = industry_df.dropna()

    # Features: Time_Index + Quarter_Num
    features = ['Time_Index', 'Quarter_Num']
    X = industry_df[features].values
    y = industry_df['target'].values

    # Train the model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(X, y)

    # If no target year/quarter is given, predict the next quarter in the dataset
    if target_year is None:
        last_row = industry_df.iloc[-1]
        next_quarter_num = (last_row['Quarter_Num'] % 4) + 1  # Wrap around Q4 → Q1
        next_time_index = last_row['Time_Index'] + 1
        input_features = np.array([[next_time_index, next_quarter_num]])
    else:
        # Calculate time index for the target year/quarter
        last_year = industry_df['Year'].max()
        last_quarter = industry_df[industry_df['Year'] == last_year]['Quarter_Num'].max()
        quarters_ahead = (target_year - last_year) * 4 + (quarter_map[target_quarter] - last_quarter)
        next_time_index = industry_df['Time_Index'].max() + quarters_ahead
        input_features = np.array([[next_time_index, quarter_map[target_quarter]]])


    prediction = model.predict(input_features)[0]
    return prediction


