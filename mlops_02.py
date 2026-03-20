import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Load and preprocess data from TSA_Example
# Note: Ensure PJME_hourly.csv is available in your working directory
df = pd.read_csv('PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

def create_features(df, combination_type='basic'):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    if combination_type == 'lags':
        df['lag_24h'] = df['PJME_MW'].shift(24)
        df['lag_1w'] = df['PJME_MW'].shift(168)
    elif combination_type == 'rolling':
        df['rolling_mean_24h'] = df['PJME_MW'].shift(1).rolling(window=24).mean()
        
    return df.dropna()

# Experiment Configurations
models = {
    'LinearRegression': LinearRegression,
    'XGBoost': xgb.XGBRegressor,
    'Ridge': Ridge
}

feature_sets = ['basic', 'lags', 'rolling']
splits = [('2015-01-01', '2016-01-01'), ('2016-01-01', '2017-01-01'), ('2017-01-01', '2018-01-01')]
hyperparams = [0.01, 0.1, 0.5] # Learning rates for XGB, Alpha for Ridge

mlflow.set_experiment("Energy_Consumption_Tracking")

for model_name, model_class in models.items():
    with mlflow.start_run(run_name=model_name, nested=True):
        for f_set in feature_sets:
            df_feat = create_features(df, combination_type=f_set)
            
            for split_date in splits:
                train = df_feat.loc[df_feat.index < split_date[0]]
                test = df_feat.loc[(df_feat.index >= split_date[0]) & (df_feat.index < split_date[1])]
                
                X_train, y_train = train.drop('PJME_MW', axis=1), train['PJME_MW']
                X_test, y_test = test.drop('PJME_MW', axis=1), test['PJME_MW']
                
                for hp in hyperparams:
                    with mlflow.start_run(run_name=f"{f_set}_{split_date[0]}_hp{hp}", nested=True):
                        # Initialize and train model
                        if model_name == 'XGBoost':
                            model = model_class(learning_rate=hp, n_estimators=100)
                        elif model_name == 'Ridge':
                            model = model_class(alpha=hp)
                        else:
                            model = model_class()
                            
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        
                        # Calculate Metrics
                        rmse = np.sqrt(mean_squared_error(y_test, preds))
                        mae = mean_absolute_error(y_test, preds)
                        
                        # Logging to MLFlow
                        mlflow.log_param("model_type", model_name)
                        mlflow.log_param("feature_combination", f_set)
                        mlflow.log_param("hyperparameter", hp)
                        mlflow.log_param("train_test_split", split_date[0])
                        mlflow.log_metric("RMSE", rmse)
                        mlflow.log_metric("MAE", mae)
                        
                        # Create and Log Plots
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(y_test.index[:168], y_test[:168], label='Actual')
                        ax.plot(y_test.index[:168], preds[:168], label='Predicted')
                        ax.set_title("Actual vs Predicted (1 Week)")
                        plt.legend()
                        fig.savefig("week_plot.png")
                        mlflow.log_artifact("week_plot.png")

                        # Create and Log Plots
                        fig, ax = plt.subplots(figsize=(15, 5))
                        ax.plot(y_test.index[:720], y_test[:720], label='Actual')
                        ax.plot(y_test.index[:720], preds[:720], label='Predicted')
                        ax.set_title("Actual vs Predicted (1 Month)")
                        plt.legend()
                        fig.savefig("month_plot.png")
                        mlflow.log_artifact("month_plot.png")
                        plt.close(fig)