import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from ta.trend import SMAIndicator, MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os
import pickle
from tqdm import tqdm

def clean_feature_names(features):
    """Clean feature names to be compatible with LightGBM"""
    cleaned = []
    for f in features:
        if isinstance(f, tuple):
            f = '_'.join(str(x) for x in f if str(x) != '')
        f = str(f).replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_')
        cleaned.append(f)
    return cleaned

def calculate_technical_indicators(data):
    """Calculate technical indicators with robust error handling"""
    try:
        data = data.copy()
        close_col = 'Close'
        
        if 'Close' not in data.columns:
            close_candidates = [col for col in data.columns if 'Close' in col]
            if close_candidates:
                close_col = close_candidates[0]
                data = data.rename(columns={close_col: 'Close'})
            else:
                raise KeyError("No 'Close' column found in data")

        close_series = data['Close'].squeeze()
        has_volume = 'Volume' in data.columns
        has_high_low = 'High' in data.columns and 'Low' in data.columns

        # Trend Indicators
        data['SMA_5'] = SMAIndicator(close=close_series, window=5).sma_indicator()
        data['SMA_10'] = SMAIndicator(close=close_series, window=10).sma_indicator()
        data['EMA_20'] = EMAIndicator(close=close_series, window=20).ema_indicator()
        
        # Momentum Indicators
        data['RSI_14'] = RSIIndicator(close=close_series, window=14).rsi()
        macd = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        
        # Volatility Indicators
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
        
        data.columns = clean_feature_names(data.columns)
        return data.dropna()
    
    except Exception as e:
        print(f"Error calculating technical indicators: {str(e)}")
        return pd.DataFrame()

def create_target(data):
    """Create target variable with robust error handling"""
    try:
        data = data.copy()
        close_col = 'Close'
        
        if 'Close' not in data.columns:
            close_candidates = [col for col in data.columns if 'Close' in col]
            if close_candidates:
                close_col = close_candidates[0]
                data = data.rename(columns={close_col: 'Close'})
            else:
                raise KeyError("No 'Close' column found in data")

        close_series = data['Close'].squeeze()
        data['Return'] = close_series.pct_change()
        data['Target'] = (data['Return'].shift(-1) > 0).astype(int)
        data.columns = clean_feature_names(data.columns)
        return data.dropna()
    
    except Exception as e:
        print(f"Error creating target: {str(e)}")
        return pd.DataFrame()

def optimize_portfolio(returns, lambda_val):
    """Portfolio optimization with Mean-Gini approach"""
    try:
        if returns.empty or len(returns) < 10:
            raise ValueError("Insufficient data for optimization")
            
        n_assets = returns.shape[1]
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        def objective(weights, mean_returns, cov_matrix, lambda_val):
            portfolio_return = np.sum(mean_returns * weights) * 52
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 52, weights)))
            Rmax = max(mean_returns.max() * 52, 1e-6)
            Gmax = max(np.sqrt(cov_matrix.max().max() * 52), 1e-6)
            return -(lambda_val * (portfolio_return/Rmax) - (1-lambda_val) * (portfolio_risk/Gmax))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.05, 0.8) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, initial_guess, 
                         args=(mean_returns, cov_matrix, lambda_val),
                         method='SLSQP', 
                         bounds=bounds, 
                         constraints=constraints)
        
        return result.x if result.success else np.array([1/n_assets] * n_assets)
    
    except Exception as e:
        print(f"Optimization error: {str(e)}")
        n_assets = returns.shape[1] if not returns.empty else 1
        return np.array([1/n_assets] * n_assets)

def markowitz_optimize_portfolio(returns, risk_aversion=1):
    """Classic Markowitz portfolio optimization"""
    try:
        if returns.empty or len(returns) < 10:
            raise ValueError("Insufficient data for optimization")
            
        n_assets = returns.shape[1]
        mean_returns = returns.mean() * 52
        cov_matrix = returns.cov() * 52

        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - risk_aversion * portfolio_risk)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.05, 0.8) for _ in range(n_assets))
        initial_guess = np.array([1/n_assets] * n_assets)
        
        result = minimize(objective, initial_guess,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        
        return result.x if result.success else np.array([1/n_assets] * n_assets)
    
    except Exception as e:
        print(f"Markowitz optimization error: {str(e)}")
        n_assets = returns.shape[1] if not returns.empty else 1
        return np.array([1/n_assets] * n_assets)

def find_best_params(model, param_grid, X_train, y_train):
    """Hyperparameter tuning with robust error handling"""
    try:
        if len(X_train) < 10 or len(y_train) < 10:
            return model.get_params()
            
        X_train.columns = clean_feature_names(X_train.columns)
        
        n_splits = min(5, max(2, len(X_train) // 3))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=tscv, 
            scoring='roc_auc', 
            n_jobs=-1,
            error_score='raise'
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_
    
    except Exception as e:
        print(f"Parameter search error: {str(e)}")
        return model.get_params()

def portfolio_rebalance(index_data, assets_data, strategy, model=None, param_grid=None, 
                      lambda_val=None, test_start_date='2010-01-01', common_index=None, 
                      index_name='ibovespa', initial_capital=100000):
    """Main portfolio rebalancing function with robust error handling"""
    
    # Configuration and directory setup
    results_dir = f"results_{index_name.lower()}"
    results_file = f"{results_dir}/{strategy.lower().replace(' ', '_')}.pkl"
    os.makedirs(results_dir, exist_ok=True)
    transaction_cost = 0.005  # 0.5% transaction cost

    try:
        # Load existing results if available
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                loaded_data = pickle.load(f)
                if len(loaded_data) == 4:
                    results_df, weights_history, prob_history, capital_history = loaded_data
                else:
                    results_df, weights_history, prob_history = loaded_data
                    capital_history = []
            
            last_date = results_df['Index'].max() if not results_df.empty else None
            if last_date and last_date >= assets_data.index[-1]:
                return results_df, weights_history, prob_history, capital_history
            else:
                start_idx = common_index.get_loc(last_date) + 1 if last_date else None
        else:
            start_idx = None
            prob_history = []
            capital_history = []

        # Data preparation with validation
        if index_data.empty or assets_data.empty:
            raise ValueError("Empty input data provided")
            
        if 'Close' not in index_data.columns:
            close_candidates = [col for col in index_data.columns if 'Close' in col]
            if close_candidates:
                index_data = index_data.rename(columns={close_candidates[0]: 'Close'})
            else:
                raise KeyError("No 'Close' column found in index data")

        # Calculate indicators and validate
        index_data_transformed = calculate_technical_indicators(index_data.copy())
        index_data_transformed = create_target(index_data_transformed)
        if index_data_transformed.empty:
            raise ValueError("Technical indicators calculation failed")
            
        assets_returns = assets_data.pct_change().dropna()
        if assets_returns.empty:
            raise ValueError("Asset returns calculation failed")
            
        common_index = index_data_transformed.index.intersection(assets_returns.index)
        if len(common_index) == 0:
            raise ValueError("No common dates between index and assets data")

        # Test period setup
        test_start_date = pd.Timestamp(test_start_date)
        pre_test_data = common_index[common_index < test_start_date]
        if len(pre_test_data) < 10:
            raise ValueError(f"Insufficient pre-test data: {len(pre_test_data)} samples")

        test_start_idx = assets_returns.index.get_loc(
            assets_returns.index[assets_returns.index >= test_start_date][0]
        )
        initial_train_size = max(test_start_idx, 10)

        if start_idx is None:
            start_idx = initial_train_size
            results = []
            weights_history = []
            prob_history = []
            capital_history = []

        # Initialize portfolio tracking
        current_capital = initial_capital
        current_positions = pd.Series(0, index=assets_data.columns)  # Positions in units
        trades_count = 0

        # Strategy execution
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

            for i in tqdm(range(start_idx, min(len(X), len(assets_returns)) - 1), 
                         desc=f"Rebalancing ({strategy})"):
                try:
                    # Prepare data
                    current_date = common_index[i]
                    next_date = common_index[i+1]
                    current_prices = assets_data.loc[current_date]
                    next_prices = assets_data.loc[next_date]
                    
                    # Calculate current portfolio value
                    current_portfolio_value = (current_positions * current_prices).sum() + current_capital
                    
                    # Model prediction
                    X_train = X.iloc[:i]
                    y_train = y.iloc[:i]
                    X_test = X.iloc[i:i+1]
                    
                    X_train.columns = clean_feature_names(X_train.columns)
                    X_test.columns = clean_feature_names(X_test.columns)
                    
                    best_model.fit(X_train, y_train)
                    prob_up = best_model.predict_proba(X_test)[0][1]
                    
                    # Portfolio optimization
                    train_returns = assets_returns.iloc[max(0, i-52*3):i]  # 3 years lookback
                    weights = optimize_portfolio(train_returns, 1 - prob_up)
                    
                    # Portfolio rebalancing
                    target_values = current_portfolio_value * weights
                    target_positions = target_values / current_prices
                    
                    trades = target_positions - current_positions
                    trade_costs = np.abs(trades) * current_prices * transaction_cost
                    total_cost = trade_costs.sum()
                    
                    # Update positions and capital
                    current_positions = target_positions
                    current_capital -= total_cost
                    trades_count += np.sum(trades != 0)
                    
                    # Calculate next portfolio value
                    next_portfolio_value = (current_positions * next_prices).sum() + current_capital
                    portfolio_return = (next_portfolio_value / current_portfolio_value) - 1
                    
                    # Record results
                    results.append({
                        'Index': next_date,
                        'Portfolio_Return': portfolio_return,
                        'Capital': next_portfolio_value
                    })
                    weights_history.append(weights)
                    prob_history.append(prob_up)
                    capital_history.append(next_portfolio_value)
                
                except Exception as e:
                    print(f"Error in iteration {i}: {str(e)}")
                    continue

            # Create results DataFrame
            results_df = pd.DataFrame(results)

        # Other strategies (Markowitz, Fixed Lambda, Buy and Hold)
        elif strategy in ['Markowitz', 'Fixed Lambda 0.1', 'Fixed Lambda 0.5', 'Fixed Lambda 1.0', 'Buy and Hold']:
            assets_returns = assets_returns.loc[common_index]
            
            for i in tqdm(range(start_idx, min(len(assets_returns), len(assets_data)) - 1), 
                         desc=f"Rebalancing ({strategy})"):
                try:
                    current_date = common_index[i]
                    next_date = common_index[i+1]
                    current_prices = assets_data.loc[current_date]
                    next_prices = assets_data.loc[next_date]
                    
                    # Calculate current portfolio value
                    current_portfolio_value = (current_positions * current_prices).sum() + current_capital
                    
                    # Get weights based on strategy
                    if strategy == 'Markowitz':
                        train_returns = assets_returns.iloc[max(0, i-52*3):i]
                        weights = markowitz_optimize_portfolio(train_returns)
                    elif 'Fixed Lambda' in strategy:
                        train_returns = assets_returns.iloc[max(0, i-52*3):i]  # 6 months lookback
                        weights = optimize_portfolio(train_returns, lambda_val)
                    elif strategy == 'Buy and Hold':
                        if i == start_idx:  # Only buy at first iteration
                            weights = np.array([1/len(assets_data.columns)] * len(assets_data.columns))
                        else:
                            weights = None
                    
                    # Portfolio rebalancing
                    if weights is not None:
                        target_values = current_portfolio_value * weights
                        target_positions = target_values / current_prices
                        
                        trades = target_positions - current_positions
                        trade_costs = np.abs(trades) * current_prices * transaction_cost
                        total_cost = trade_costs.sum()
                        
                        current_positions = target_positions
                        current_capital -= total_cost
                        trades_count += np.sum(trades != 0)
                    
                    # Calculate next portfolio value
                    next_portfolio_value = (current_positions * next_prices).sum() + current_capital
                    portfolio_return = (next_portfolio_value / current_portfolio_value) - 1
                    
                    # Record results
                    results.append({
                        'Index': next_date,
                        'Portfolio_Return': portfolio_return,
                        'Capital': next_portfolio_value
                    })
                    weights_history.append(weights if weights is not None else weights_history[-1])
                    prob_history.append(lambda_val if 'Fixed Lambda' in strategy else None)
                    capital_history.append(next_portfolio_value)
                
                except Exception as e:
                    print(f"Error in iteration {i}: {str(e)}")
                    continue

            # Create results DataFrame
            results_df = pd.DataFrame(results)

        # Index strategy (just track the index)
        elif strategy == 'Ibovespa':
            index_returns = index_data_transformed['Close'].pct_change().dropna()
            valid_index = index_returns.index.intersection(common_index)
            test_returns = index_returns.loc[valid_index]
            test_returns = test_returns[test_returns.index >= test_start_date]
            
            if start_idx != initial_train_size:
                test_returns = test_returns.iloc[start_idx - initial_train_size:]
            
            results_df = pd.DataFrame({
                'Index': test_returns.index,
                'Portfolio_Return': test_returns.values,
                'Capital': initial_capital * (1 + test_returns).cumprod()
            })
            weights_history = [np.array([1.0] + [0.0]*(len(assets_data.columns)-1))] * len(results_df)
            prob_history = [None] * len(results_df)
            capital_history = results_df['Capital'].tolist()

        # Add trades count to results
        results_df['Trades_Count'] = trades_count

        # Save results
        with open(results_file, 'wb') as f:
            pickle.dump((results_df, weights_history, prob_history, capital_history), f)

        return results_df, weights_history, prob_history, capital_history

    except Exception as e:
        print(f"Error in portfolio_rebalance: {str(e)}")
        # Return empty results if error occurs
        empty_df = pd.DataFrame(columns=['Index', 'Portfolio_Return', 'Capital', 'Trades_Count'])
        return empty_df, [], [], []

def calculate_metrics(results_df, initial_capital=100000):
    """Calculate performance metrics with robust error handling"""
    try:
        if results_df.empty:
            return {
                'Capital Inicial': initial_capital,
                'Capital Final': initial_capital,
                'Retorno Cumulativo (%)': 0.0,
                'Retorno Anualizado (%)': 0.0,
                'Volatilidade Anualizada (%)': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown (%)': 0.0,
                'Dias Positivos (%)': 0.0,
                'Retorno Médio Diário (%)': 0.0,
                'Retorno Médio Dias Positivos (%)': 0.0,
                'Retorno Médio Dias Negativos (%)': 0.0,
                'Fator de Lucro': 1.0,
                'Número de Transações': 0
            }
        
        # Calculate returns based on available data
        if 'Capital' in results_df.columns and not results_df['Capital'].empty:
            results_df['Cumulative_Return'] = (results_df['Capital'] / initial_capital) - 1
            results_df['Daily_Return'] = results_df['Capital'].pct_change()
        else:
            results_df['Cumulative_Return'] = (1 + results_df['Portfolio_Return']).cumprod() - 1
            results_df['Daily_Return'] = results_df['Portfolio_Return']
            results_df['Capital'] = initial_capital * (1 + results_df['Cumulative_Return'])
        
        daily_returns = results_df['Daily_Return'].dropna()
        if len(daily_returns) == 0:
            raise ValueError("No valid returns to calculate metrics")
        
        # Basic metrics
        cumulative_return = results_df['Cumulative_Return'].iloc[-1] if not results_df.empty else 0
        annualized_return = (1 + cumulative_return) ** (52 / len(results_df)) - 1 if len(results_df) > 0 else 0
        annualized_volatility = daily_returns.std() * np.sqrt(52) if len(daily_returns) > 1 else 0
        sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility != 0 else 0
        
        # Drawdown calculation
        cumulative_returns = 1 + results_df['Cumulative_Return']
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        # Other metrics
        winning_days = (daily_returns > 0).mean() if not daily_returns.empty else 0
        avg_daily_return = daily_returns.mean() if not daily_returns.empty else 0
        avg_winning_return = daily_returns[daily_returns > 0].mean() if len(daily_returns[daily_returns > 0]) > 0 else 0
        avg_losing_return = daily_returns[daily_returns <= 0].mean() if len(daily_returns[daily_returns <= 0]) > 0 else 0
        profit_factor = -avg_winning_return / avg_losing_return if avg_losing_return != 0 else np.nan
        
        # Number of transactions
        trades_count = results_df['Trades_Count'].iloc[-1] if 'Trades_Count' in results_df.columns else 0
        
        return {
            'Capital Inicial': initial_capital,
            'Capital Final': results_df['Capital'].iloc[-1] if not results_df.empty else initial_capital,
            'Retorno Cumulativo (%)': cumulative_return * 100,
            'Retorno Anualizado (%)': annualized_return * 100,
            'Volatilidade Anualizada (%)': annualized_volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Dias Positivos (%)': winning_days * 100,
            'Retorno Médio Diário (%)': avg_daily_return * 100,
            'Retorno Médio Dias Positivos (%)': avg_winning_return * 100,
            'Retorno Médio Dias Negativos (%)': avg_losing_return * 100,
            'Fator de Lucro': profit_factor,
            'Número de Transações': trades_count
        }
    
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'Capital Inicial': initial_capital,
            'Capital Final': initial_capital,
            'Retorno Cumulativo (%)': 0.0,
            'Retorno Anualizado (%)': 0.0,
            'Volatilidade Anualizada (%)': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown (%)': 0.0,
            'Dias Positivos (%)': 0.0,
            'Retorno Médio Diário (%)': 0.0,
            'Fator de Lucro': 1.0,
            'Número de Transações': 0
        }