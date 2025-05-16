#Time Series Analysis*
###############################
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

###############################



#ADF test
def adf_test(randomwalk):
    from statsmodels.tsa.stattools import adfuller
    ADF_result = adfuller(random_walk)
    print(f'ADF Statistic: {ADF_result[0]}')
    print(f'p-value: {ADF_result[1]}')


#Mean absolute error
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#Plotting ACF
def plot_acf_pacf(data,lags,plot_type):
    from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
    import matplotlib.pyplot as plt
    
    if plot_type='acf':
        plot_acf(data, lags=lags)
        plt.tight_layout()
    else:
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
def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm_notebook(order_list):
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
def optimize_SARIMA(endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int) -> pd.DataFrame:
    
    results = []
    
    for order in tqdm_notebook(order_list):
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
def best_model_combo(model_type, combo_df, train, test, d, start, end, D=None, s=None):
    """
    Finds the best (p, q) or (p, q, P, Q) combination based on lowest MAPE.
    
    Parameters:
        model_type (str): 'ARIMA' or 'SARIMA'
        combo_df (pd.DataFrame): DataFrame with column ['(p,q,P,Q)' or '(p,q)'] as tuples
        train (array-like or pd.Series): Training data
        test (array-like or pd.Series): Test data (actual values)
        d (int): Non-seasonal differencing order
        start (int or datetime): Start index for prediction
        end (int or datetime): End index for prediction
        D (int or None): Seasonal differencing order (for SARIMA only)
        s (int or None): Seasonal period (for SARIMA only)

    Returns:
        dict: Best combination and corresponding MAPE
    """
    lowest_mape = float('inf')
    best_combo = None

    # Validate model type
    if model_type not in ['ARIMA', 'SARIMA']:
        raise ValueError("model_type must be 'ARIMA' or 'SARIMA'")

    for index, row in combo_df.iterrows():
        # Extract parameters based on model type
        if model_type == 'ARIMA':
            p, q = row['(p,q)']
            P, Q = 0, 0  # No seasonal part
        else:
            p, q, P, Q = row['(p,q,P,Q)']

        try:
            # Fit the appropriate model
            if model_type == 'ARIMA':
                model = SARIMAX(train, order=(p, d, q))
            else:
                if D is None or s is None:
                    raise ValueError("For SARIMA, 'D' and 's' must be provided.")
                model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))

            model_fit = model.fit(disp=False)

            # Get predictions
            pred = model_fit.get_prediction(start=start, end=end).predicted_mean

            # Calculate MAPE
            current_mape = mape(test, pred)

            # Update best combo if better
            if current_mape < lowest_mape:
                lowest_mape = current_mape
                best_combo = (p, q, P, Q) if model_type == 'SARIMA' else (p, q)

        except Exception as e:
            print(f"Error fitting model with {row['(p,q,P,Q)' if model_type == 'SARIMA' else '(p,q)']}: {e}")
            continue

    return {
        'Best Combination': best_combo,
        'Lowest MAPE': lowest_mape
    }


def print_SARIMAX_results(data,order:tuple):
    model = SARIMAX(data, order=order, simple_differencing=False)
    model_fit = model.fit(disp=False)
    print(model_fit.summary())

def Lljunbox(model):
    residuals = model.resid
    lbvalue, pvalue = acorr_ljungbox(residuals, np.arange(1, 11, 1))
    print(pvalue)


def recursive_forecast(endog: Union[pd.Series, list], exog: Union[pd.Series, list], train_len: int, horizon: int, window: int, method: str) -> list:
    
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
            model = SARIMAX(endog[:i], exog[:i], order=(3,1,3), seasonal_order=(0,0,0,4), simple_differencing=False)
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_SARIMAX.extend(oos_pred)
            
        return pred_SARIMAX





