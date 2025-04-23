
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # Fixed import
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def load_and_prepare_data():
    
    data = pd.read_csv("data/2124.csv").apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    df = pd.concat([data], ignore_index=True)
    df.columns = df.columns.str.strip()

    
    df_long = df.melt(id_vars=['Description'],
                      value_vars=[f'{year}:Q{q}' for year in range(2021,2025) for q in range(1,5)],
                      var_name='Quarter',
                      value_name='Value')
    
    #  sort
    df_long[['Year', 'Quarter_Label']] = df_long['Quarter'].str.split(':', expand=True)
    df_long['Year'] = df_long['Year'].astype(int)
    df_long['Datetime'] = pd.to_datetime(
        df_long['Year'].astype(str) + ' ' + df_long['Quarter_Label'].str.replace('Q', '')
    )
    return df_long.sort_values(['Description', 'Datetime']).rename(columns={'Value':'GDP'})
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def forecast_and_evaluate(industry_name='All industry total'):
    df_long = load_and_prepare_data()
    
    # Filter and prepare industry data
    industry_df = df_long[df_long['Description'].str.lower() == industry_name.lower()].copy()
    

    if len(industry_df) == 0:
        raise ValueError(f"No data found for industry: {industry_name}")
    
    industry_df = industry_df.sort_values('Datetime')
    
    #  lag validation
    for lag in [1,2,3,4]:
        industry_df[f'GDP_lag{lag}'] = industry_df['GDP'].shift(lag)
    
    industry_df['Quarter_Num'] = industry_df['Quarter_Label'].map({'Q1':1, 'Q2':2, 'Q3':3, 'Q4':4})
    industry_df['target'] = industry_df['GDP'].shift(-1)
    
    
    industry_df = industry_df.dropna()
    
    
    # historical data
    if len(industry_df) < 11:  
        raise ValueError(f"Insufficient data points. Found {len(industry_df)} rows, need minimum 8")

    # Temporal split with datetime validation
    train_mask = industry_df['Datetime'] < '2024-01-12'
    test_mask = industry_df['Datetime'] >= '2024-01-12'
    
   
    X_train = industry_df[train_mask][['GDP_lag1', 'GDP_lag2', 'GDP_lag3', 'GDP_lag4', 'Quarter_Num']]
    X_test = industry_df[test_mask][['GDP_lag1', 'GDP_lag2', 'GDP_lag3', 'GDP_lag4', 'Quarter_Num']]
    
    if len(X_train) == 0:
        raise ValueError("Training data empty - check your date ranges or data filtering")
    if len(X_test) == 0:
        print("⚠️ No test data found - using last 4 quarters as test set")
        X_test = industry_df.iloc[-4:][['GDP_lag1', 'GDP_lag2', 'GDP_lag3', 'GDP_lag4', 'Quarter_Num']]
    
    # Feature/target assignment
    y_train = industry_df[train_mask]['target']
    y_test = industry_df[test_mask]['target']
    
    
    if X_train.shape[1] != 5:  # 5 features expected
        raise ValueError(f"Feature dimension mismatch. Expected 5 features, got {X_train.shape[1]}")
    
    # Scaling with safety checks
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except ValueError as e:
        print("Scaling error - data validation failed")
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        raise

   
    model = XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.09,
        subsample=0.5,
        colsample_bytree=0.5,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )
    
    # Evaluation
   
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    
    #  prediction
    last_row = industry_df.iloc[-1][['GDP_lag1', 'GDP_lag2', 'GDP_lag3', 'GDP_lag4', 'Quarter_Num']]
    next_q = (last_row['Quarter_Num'] % 4) + 1
    new_lags = [industry_df['GDP'].iloc[-1]] + last_row[['GDP_lag1','GDP_lag2','GDP_lag3']].tolist()
    
    X_future = pd.DataFrame([new_lags + [next_q]], columns=['GDP_lag1', 'GDP_lag2', 'GDP_lag3', 'GDP_lag4', 'Quarter_Num'])
    X_future_scaled = scaler.transform(X_future)
    prediction = model.predict(X_future_scaled)[0]
    
    print(f"Model Evaluation for {industry_name}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Predicted Next Quarter GDP: {prediction:.2f}")

    
    return prediction
