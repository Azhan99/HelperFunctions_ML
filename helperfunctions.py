###############################
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller,grangercausalitytests
from tqdm import tqdm
from typing import Union
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import mean_squared_error


#ADF test
def adf_test(random_walk):
    ADF_result = adfuller(random_walk)
    print(f'ADF Statistic: {ADF_result[0]}')
    print(f'p-value: {ADF_result[1]}')


#Mean absolute error
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Plotting ACF
def plot_acf_pacf(data,lags):
    plot_acf(data, lags=lags)
    plt.tight_layout()
    plot_pacf(data, lags=lags)
    plt.tight_layout()


#Rolling Forecast
def rolling_forecast(df: pd.DataFrame, train_len: int, horizon: int, window: int, method: str,p:int,d:int,q:int) -> list:
    total_len = train_len + horizon
    
    if method == 'mean':
        pred_mean = []
        
        for i in range(train_len, total_len, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        
        return pred_mean[:horizon]  # Trim to match test length

    elif method == 'last':
        pred_last_value = []
        
        for i in range(train_len, total_len, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
        
        return pred_last_value[:horizon]  # Trim to match test length
    
    elif method == 'MA':
        pred_MA = []
        
        for i in range(train_len, total_len, window):
            model = SARIMAX(df[:i], order=(p, d, q))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_MA.extend(oos_pred)
        
        return pred_MA[:horizon]  # Trim to match test length

#Optimize_ARMA
def optimize_arima(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm(order_list):
        try: 
            model = SARIMAX(endog, order=(order[0], d, order[1]), simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df

#Optimize_SARIMA
def optimize_sarima(endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm(order_list):
        try: 
            model = SARIMAX(
                endog, 
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


##Best Combination
def best_model_combo(model_type, combo_df, train, test, d, exog_train=None, exog_test=None, D=None, s=None):
    """
    Finds the best (p, q) or (p, q, P, Q) combination based on lowest RMSE,
    while ensuring all Ljung-Box test p-values > 0.05 for residuals.
    
    Parameters:
        model_type (str): 'ARIMA', 'SARIMA', or 'SARIMAX'
        combo_df (pd.DataFrame): DataFrame with column ['(p,q,P,Q)' or '(p,q)'] as tuples
        train (array-like or pd.Series): Training data
        test (array-like or pd.Series): Test data (actual values)
        d (int): Non-seasonal differencing order
        exog_train (pd.DataFrame): Training data for exogenous variables (for SARIMAX only)
        exog_test (pd.DataFrame): Test data for exogenous variables (for SARIMAX only)
        D (int or None): Seasonal differencing order (for SARIMA/SARIMAX only)
        s (int or None): Seasonal period (for SARIMA/SARIMAX only)

    Returns:
        dict: Best valid combination and corresponding RMSE
    """

    lowest_rmse = float('inf')
    best_combo = None
    valid_combos = []

    if model_type not in ['ARIMA', 'SARIMA', 'SARIMAX']:
        raise ValueError("model_type must be 'ARIMA', 'SARIMA', or 'SARIMAX'")

    # Validate exog parameters for SARIMAX
    if model_type == 'SARIMAX':
        if exog_train is None or exog_test is None:
            raise ValueError("For SARIMAX, both 'exog_train' and 'exog_test' must be provided.")
        if D is None or s is None:
            raise ValueError("For SARIMAX, 'D' and 's' must be provided.")

    for _, row in combo_df.iterrows():
        try:
            # Extract parameters based on model type
            if model_type == 'ARIMA':
                p, q = row['(p,q)']
                P, Q = 0, 0
                seasonal_order = (0, 0, 0, 0)  # No seasonal component
                
            else:  # SARIMA or SARIMAX
                p, q, P, Q = row['(p,q,P,Q)']
                if D is None or s is None:
                    raise ValueError("For SARIMA/SARIMAX, 'D' and 's' must be provided.")
                seasonal_order = (P, D, Q, s)
                
            # Fit the model
            if model_type == 'SARIMAX':
                model = SARIMAX(train, exog=exog_train, order=(p, d, q), 
                               seasonal_order=seasonal_order, simple_differencing=False,
                               enforce_stationarity=False, enforce_invertibility=False)
            else:
                model = SARIMAX(train, order=(p, d, q), seasonal_order=seasonal_order,
                               simple_differencing=False, enforce_stationarity=False, 
                               enforce_invertibility=False)
                
            model_fit = model.fit(disp=False)

            # Check residuals using Ljung-Box test
            residuals = model_fit.resid
            lb_test = acorr_ljungbox(residuals, lags=np.arange(1, 11, 1), return_df=True)

            # Ensure all p-values > 0.05
            if (lb_test['lb_pvalue'] > 0.05).all():
                # Make predictions using get_forecast for out-of-sample predictions
                if model_type == 'SARIMAX':
                    forecast = model_fit.get_forecast(steps=len(test), exog=exog_test)
                    pred = forecast.predicted_mean
                else:
                    forecast = model_fit.get_forecast(steps=len(test))
                    pred = forecast.predicted_mean

                # Compute RMSE
                current_rmse = np.sqrt(mean_squared_error(test, pred))

                # Store valid combo
                valid_combos.append({
                    'combo': (p, q, P, Q) if model_type in ['SARIMA', 'SARIMAX'] else (p, q),
                    'rmse': current_rmse
                })

                # Update best combo
                if current_rmse < lowest_rmse:
                    lowest_rmse = current_rmse
                    best_combo = (p, q, P, Q) if model_type in ['SARIMA', 'SARIMAX'] else (p, q)

        except Exception as e:
            print(f"Error fitting model {row['(p,q,P,Q)' if model_type in ['SARIMA', 'SARIMAX'] else '(p,q)']}: {e}")
            continue

    if best_combo is None:
        print("No combinations passed the Ljung-Box test (all p-values < 0.05)")
        return None

    return {
        'Best Combination': best_combo,
        'Lowest RMSE': lowest_rmse,
        'Valid Combinations': valid_combos
    }



def print_sarima_results(data,order:tuple):
    model = SARIMAX(data, order=order, simple_differencing=False)
    model_fit = model.fit(disp=False)
    print(model_fit.summary())

def Lljunbox(model):
    residuals = model.resid
    lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))
    print(pvalue)


##SARIMAX function
def optimize_sarimax(endog: Union[pd.Series, list], exog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm(order_list):
        try: 
            model = SARIMAX(
                endog,
                exog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue
            
        aic = model.aic
        results.append([order, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC']
    
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df



def recursive_forecast_sarimax(endog: Union[pd.Series, list], exog: Union[pd.Series, list], train_len: int, horizon: int, window: int, method: str,p:int,d:int,q:int,P:int,D:int,Q:int,m:int) -> list:
    
    total_len = train_len + horizon

    if method == 'last':
        pred_last_value = []
        
        for i in range(train_len, total_len, window):
            last_value = endog[:i].iloc[-1]
            pred_last_value.extend(last_value for _ in range(window))
            
        return pred_last_value
    
    elif method == 'SARIMAX':
        pred_SARIMAX = []
        
        for i in range(train_len, total_len, window):
            model = SARIMAX(endog[:i], exog[:i], order=(p,d,q), seasonal_order=(P,D,Q,m), simple_differencing=False)
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_SARIMAX.extend(oos_pred)
            
        return pred_SARIMAX




def optimize_var(endog: Union[pd.Series, list]) -> pd.DataFrame:
    
    results = []
    
    for i in tqdm(range(15)):
        try:
            model = VARMAX(endog, order=(i, 0)).fit(dips=False)
        except:
            continue
            
        aic = model.aic
        results.append([i, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['p', 'AIC']
    
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df



def granger_casualty_test(first_val: str, second_val: str, data: pd.DataFrame, d: int, lag: list):
    """
    Tests Granger Causality between two variables in both directions.
    
    Parameters:
        first_val (str): Name of first column
        second_val (str): Name of second column
        data (pd.DataFrame): DataFrame with both columns
        d (int): Number of differences to apply (e.g., 1)
        lag (list): List of lags to test (e.g., [2])
    """
    # Subset data and difference
    data_subset = data[[first_val, second_val]]
    data_diffed = data_subset.diff()[d:].dropna()  # Remove NaNs after differencing
    
    print(f"\n{first_val} Granger-causes {second_val}?")
    print('------------------')
    granger_1 = grangercausalitytests(data_diffed, maxlag=lag[0])
    
    print(f"\n{second_val} Granger-causes {first_val}?")
    print('------------------')
    granger_2 = grangercausalitytests(data_diffed[[second_val, first_val]], maxlag=lag[0])



def rolling_forecast_var(df: pd.DataFrame, train_len: int, horizon: int, window: int,first:str,second:str) -> list:
    
    total_len = train_len + horizon
    end_idx = train_len
    
    if method == 'VAR':

        first_pred_VAR = []
        second_pred_VAR = []
        
        for i in range(train_len, total_len, window):
            model = VARMAX(df[:i], order=(3,0))
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)
            
            oos_pred_first = predictions.predicted_mean.iloc[-window:][first]
            oos_pred_second = predictions.predicted_mean.iloc[-window:][second]
            
            realdpi_pred_VAR.extend(oos_pred_first)
            realcons_pred_VAR.extend(oos_pred_second)
        
        return first_pred_VAR, second_pred_VAR
    
    elif method == 'last':
        first_pred_last = []
        second_pred_last = []
        
        for i in range(train_len, total_len, window):
            
            first_last = df[:i].iloc[-1][first]
            second_last = df[:i].iloc[-1][second]
            
            first_pred_last.extend(first_last for _ in range(window))
            second_pred_last.extend(second_last for _ in range(window))
            
        return frist_pred_last, second_pred_last






