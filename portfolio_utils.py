import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import yfinance as yf
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
import os
import pickle
from tqdm import tqdm

def clean_feature_names(features):
    """Clean feature names to be compatible with LightGBM"""
    cleaned = []
    for f in features:
        if isinstance(f, tuple):
            # Handle MultiIndex by joining with underscores
            f = '_'.join(str(x) for x in f if str(x) != '')
        # Remove special characters
        f = str(f).replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_')
        cleaned.append(f)
    return cleaned

def calculate_technical_indicators(data):
    data = data.copy()
    close_col = 'Close'
    if 'Close' not in data.columns:
        close_candidates = [col for col in data.columns if 'Close' in col]
        if close_candidates:
            close_col = close_candidates[0]
            data = data.rename(columns={close_col: 'Close'})
        else:
            raise KeyError(f"'Close' column not found. Available columns: {data.columns.tolist()}")
    
    close_series = data['Close'].squeeze()
    has_volume = 'Volume' in data.columns
    has_high_low = 'High' in data.columns and 'Low' in data.columns

    # Calculate indicators
    data['SMA_5'] = SMAIndicator(close=close_series, window=5).sma_indicator()
    data['SMA_10'] = SMAIndicator(close=close_series, window=10).sma_indicator()
    data['RSI_14'] = RSIIndicator(close=close_series, window=14).rsi()
    
    macd = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    data['EMA_20'] = EMAIndicator(close=close_series, window=20).ema_indicator()
    
    bb = BollingerBands(close=close_series, window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['BB_Width'] = (data['BB_High'] - data['BB_Low']) / bb.bollinger_mavg()
    
    if has_high_low:
        data['ATR_14'] = AverageTrueRange(
            high=data['High'], 
            low=data['Low'], 
            close=close_series, 
            window=14
        ).average_true_range()
    
    if has_volume:
        data['OBV'] = OnBalanceVolumeIndicator(
            close=close_series, 
            volume=data['Volume']
        ).on_balance_volume()
    
    # Clean column names
    data.columns = clean_feature_names(data.columns)
    return data

def create_target(data):
    close_col = 'Close'
    if 'Close' not in data.columns:
        close_candidates = [col for col in data.columns if 'Close' in col]
        if close_candidates:
            close_col = close_candidates[0]
            data = data.rename(columns={close_col: 'Close'})
        else:
            raise KeyError(f"'Close' column not found. Available columns: {data.columns.tolist()}")
    
    close_series = data['Close'].squeeze()
    data['Return'] = close_series.pct_change()
    data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
    data.columns = clean_feature_names(data.columns)
    return data

def optimize_portfolio(returns, lambda_val):
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def objective(weights, mean_returns, cov_matrix, lambda_val):
        portfolio_return = np.sum(mean_returns * weights) * 52
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 52, weights)))
        Rmax = mean_returns.max() * 52
        Gmax = np.sqrt(cov_matrix.max().max() * 52)
        normalized_return = portfolio_return / Rmax
        normalized_risk = portfolio_risk / Gmax
        return -(lambda_val * normalized_return - (1 - lambda_val) * normalized_risk)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.8) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    result = minimize(objective, initial_guess, 
                     args=(mean_returns, cov_matrix, lambda_val),
                     method='SLSQP', 
                     bounds=bounds, 
                     constraints=constraints)
    return result.x

def markowitz_optimize_portfolio(returns, risk_aversion=1):
    n_assets = returns.shape[1]
    mean_returns = returns.mean() * 52
    cov_matrix = returns.cov() * 52

    def objective(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_aversion * portfolio_risk)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 0.8) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)
    result = minimize(objective, initial_guess, 
                     method='SLSQP', 
                     bounds=bounds, 
                     constraints=constraints)
    return result.x

def find_best_params(model, param_grid, X_train, y_train):
    n_samples = len(X_train)
    n_splits = min(5, max(2, n_samples // 2))
    if n_samples < 2:
        return model.get_params()
    
    # Ensure clean feature names for LightGBM
    X_train.columns = clean_feature_names(X_train.columns)
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=TimeSeriesSplit(n_splits=n_splits), 
        scoring='roc_auc', 
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def portfolio_rebalance(index_data, assets_data, strategy, model=None, param_grid=None, 
                      lambda_val=None, test_start_date='2010-01-01', common_index=None, 
                      index_name='sp500'):
    # Configuration
    results_dir = f"results_{index_name.lower()}"
    results_file = f"{results_dir}/{strategy.lower().replace(' ', '_')}.pkl"
    os.makedirs(results_dir, exist_ok=True)

    # Load existing results if available
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            loaded_data = pickle.load(f)
            if len(loaded_data) == 3:
                results_df, weights_history, prob_history = loaded_data
            else:
                results_df, weights_history = loaded_data
                prob_history = []
        last_date = results_df['Index'].max()
        if last_date >= assets_data.index[-1]:
            return results_df, weights_history, prob_history
        else:
            start_idx = common_index.get_loc(last_date) + 1
    else:
        start_idx = None
        prob_history = []

    # Prepare data
    if 'Close' not in index_data.columns:
        close_candidates = [col for col in index_data.columns if 'Close' in col]
        if close_candidates:
            index_data = index_data.rename(columns={close_candidates[0]: 'Close'})
        else:
            raise KeyError(f"'Close' column not found. Available columns: {index_data.columns.tolist()}")

    index_data_transformed = calculate_technical_indicators(index_data.copy())
    index_data_transformed = create_target(index_data_transformed).dropna()
    assets_returns = assets_data.pct_change().dropna()
    common_index = index_data_transformed.index.intersection(assets_returns.index) if common_index is None else common_index

    test_start_date = pd.Timestamp(test_start_date)
    pre_test_data = common_index[common_index < test_start_date]
    if len(pre_test_data) < 10:
        raise ValueError(f"Insufficient data before {test_start_date}. Samples: {len(pre_test_data)}")

    test_start_idx = assets_returns.index.get_loc(assets_returns.index[assets_returns.index >= test_start_date][0])
    initial_train_size = max(test_start_idx, 10)

    if start_idx is None:
        start_idx = initial_train_size
        results_df = pd.DataFrame()
        weights_history = []
        prob_history = []

    # ML Strategies
    if strategy in ['XGBoost', 'RandomForest', 'LightGBM', 'MLPClassifier']:
        features = ['SMA_5', 'SMA_10', 'RSI_14', 'MACD', 'MACD_Signal']
        X = index_data_transformed[features].loc[common_index]
        X.columns = clean_feature_names(X.columns)
        
        y = index_data_transformed['Target'].loc[common_index]
        assets_returns = assets_returns.loc[common_index]

        if start_idx == initial_train_size:
            X_train_initial = X.iloc[:initial_train_size]
            y_train_initial = y.iloc[:initial_train_size]
            best_params = find_best_params(model, param_grid, X_train_initial, y_train_initial)
            best_model = model.__class__(random_state=42, **best_params)
            if isinstance(best_model, LGBMClassifier):
                best_model.set_params(verbose=-1)
        else:
            X_train_initial = X.iloc[:start_idx]
            y_train_initial = y.iloc[:start_idx]
            best_params = find_best_params(model, param_grid, X_train_initial, y_train_initial)
            best_model = model.__class__(random_state=42, **best_params)
            if isinstance(best_model, LGBMClassifier):
                best_model.set_params(verbose=-1)

        portfolio_returns, new_weights_history, indices, new_prob_history = [], [], [], []
        for i in tqdm(range(start_idx, len(X) - 1), desc=f"Rebalancing ({strategy})"):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i:i+1]
            
            # Ensure clean feature names
            X_train.columns = clean_feature_names(X_train.columns)
            X_test.columns = clean_feature_names(X_test.columns)
            
            best_model.fit(X_train, y_train)
            prob_up = best_model.predict_proba(X_test)[0][1]
            train_returns = assets_returns.iloc[i-52*3:i]
            weights = optimize_portfolio(train_returns, 1 - prob_up)
            next_period_returns = assets_returns.iloc[i+1]
            portfolio_return = np.sum(next_period_returns * weights)
            
            portfolio_returns.append(portfolio_return)
            new_weights_history.append(weights)
            indices.append(X.index[i+1])
            new_prob_history.append(prob_up)

        new_results_df = pd.DataFrame({'Index': indices, 'Portfolio_Return': portfolio_returns})
        results_df = pd.concat([results_df, new_results_df], ignore_index=True)
        weights_history.extend(new_weights_history)
        prob_history.extend(new_prob_history)

    # Other strategies (Markowitz, Fixed Lambda, Buy and Hold, Index)
    elif strategy == 'Markowitz':
        assets_returns = assets_returns.loc[common_index]
        portfolio_returns, new_weights_history, indices = [], [], []
        for i in tqdm(range(start_idx, len(assets_returns) - 1), desc="Rebalancing (Markowitz)"):
            train_returns = assets_returns.iloc[i-52*3:i]
            weights = markowitz_optimize_portfolio(train_returns)
            next_period_returns = assets_returns.iloc[i+1]
            portfolio_return = np.sum(next_period_returns * weights)
            
            portfolio_returns.append(portfolio_return)
            new_weights_history.append(weights)
            indices.append(assets_returns.index[i+1])

        new_results_df = pd.DataFrame({'Index': indices, 'Portfolio_Return': portfolio_returns})
        results_df = pd.concat([results_df, new_results_df], ignore_index=True)
        weights_history.extend(new_weights_history)
        prob_history.extend([None] * len(new_weights_history))

    elif 'Fixed Lambda' in strategy:
        assets_returns = assets_returns.loc[common_index]
        portfolio_returns, new_weights_history, indices = [], [], []
        for i in tqdm(range(start_idx, len(assets_returns) - 1), desc=f"Rebalancing ({strategy})"):
            train_returns = assets_returns.iloc[i-26:i]
            weights = optimize_portfolio(train_returns, lambda_val)
            next_period_returns = assets_returns.iloc[i+1]
            portfolio_return = np.sum(next_period_returns * weights)
            
            portfolio_returns.append(portfolio_return)
            new_weights_history.append(weights)
            indices.append(assets_returns.index[i+1])

        new_results_df = pd.DataFrame({'Index': indices, 'Portfolio_Return': portfolio_returns})
        results_df = pd.concat([results_df, new_results_df], ignore_index=True)
        weights_history.extend(new_weights_history)
        prob_history.extend([lambda_val] * len(new_weights_history))

    elif strategy == 'Buy and Hold':
        assets_returns = assets_returns.loc[common_index]
        test_returns = assets_returns.loc[assets_returns.index >= test_start_date]
        if start_idx != initial_train_size:
            test_returns = test_returns.iloc[start_idx - initial_train_size:]
        n_assets = assets_returns.shape[1]
        weights = np.array([1/n_assets] * n_assets)
        portfolio_returns = (test_returns * weights).sum(axis=1)
        
        new_results_df = pd.DataFrame({'Index': test_returns.index, 'Portfolio_Return': portfolio_returns})
        results_df = pd.concat([results_df, new_results_df], ignore_index=True)
        if start_idx == initial_train_size:
            weights_history = [weights] * len(new_results_df)
            prob_history = [0.5] * len(new_results_df)
        else:
            weights_history.extend([weights] * len(new_results_df))
            prob_history.extend([0.5] * len(new_results_df))

    elif strategy in ['S&P 500', 'Ibovespa']:
        index_returns = index_data_transformed['Close'].pct_change().dropna()
        valid_index = index_returns.index.intersection(common_index)
        test_returns = index_returns.loc[valid_index]
        test_returns = test_returns[test_returns.index >= test_start_date]
        if start_idx != initial_train_size:
            test_returns = test_returns.iloc[start_idx - initial_train_size:]
        
        new_results_df = pd.DataFrame({'Index': test_returns.index, 'Portfolio_Return': test_returns.values})
        results_df = pd.concat([results_df, new_results_df], ignore_index=True)
        weights_history = None
        prob_history = [None] * len(new_results_df)

    # Save results
    with open(results_file, 'wb') as f:
        pickle.dump((results_df, weights_history, prob_history), f)

    return results_df, weights_history, prob_history

def calculate_metrics(results_df):
    cumulative_return = (1 + results_df['Portfolio_Return']).cumprod() - 1
    annualized_return = (cumulative_return.iloc[-1] + 1) ** (52 / len(cumulative_return)) - 1
    average_return = results_df['Portfolio_Return'].mean() * 52
    std_dev = results_df['Portfolio_Return'].std() * np.sqrt(52)
    sharpe_ratio = average_return / std_dev if std_dev != 0 else np.nan
    
    return {
        'Retorno Cumulativo (%)': cumulative_return.iloc[-1] * 100,
        'Retorno Anualizado (%)': annualized_return * 100,
        'Desvio Padrão Anualizado (%)': std_dev * 100,
        'Sharpe Ratio': sharpe_ratio
    }