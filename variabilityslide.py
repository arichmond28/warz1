import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Function to calculate basic statistics
def calculate_statistics(prices):
    stats = np.array([
        np.mean(prices),
        np.median(prices),
        np.std(prices),
        np.min(prices),
        np.max(prices),
        np.max(prices) - np.min(prices),
        np.sum(prices),
        skew(prices),
        kurtosis(prices)
    ])
    return stats

# Function to calculate price movement statistics
def calculate_price_movement(open_prices, close_prices):
    absolute_change = close_prices[-1] - open_prices[0]
    percent_change = (absolute_change / open_prices[0]) * 100
    return np.array([absolute_change, percent_change])

# Function to calculate stock price volatility using standard deviation of log returns
def calculate_volatility(prices, annualized=True, periods_per_year=252):
    log_returns = np.diff(np.log(prices))
    volatility = np.std(log_returns)
    if annualized.all():
        volatility *= np.sqrt(periods_per_year)
    return volatility

# Function to calculate moving averages
def calculate_moving_average(prices, window_size):
    moving_average = np.convolve(prices, np.ones(window_size)/window_size, mode='valid')
    return moving_average

# Function to calculate RSI
def calculate_rsi(prices, window_size=14):
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gain[-window_size:])
    avg_loss = np.mean(loss[-window_size:])
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return np.array([rsi])

# Function to calculate MACD
def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    # Calculate short and long-term exponential moving averages (EMA)
    short_ema = calculate_moving_average(prices, short_window)
    long_ema = calculate_moving_average(prices, long_window)
    
    # Align the lengths of short_ema and long_ema by trimming to the same length
    min_length = min(len(short_ema), len(long_ema))
    short_ema = short_ema[-min_length:]
    long_ema = long_ema[-min_length:]
    
    # Calculate the MACD line
    macd = short_ema - long_ema
    
    # Calculate the signal line (EMA of the MACD line)
    signal_line = calculate_moving_average(macd, signal_window)
    
    # Align lengths of MACD and signal_line
    min_length_macd_signal = min(len(macd), len(signal_line))
    macd = macd[-min_length_macd_signal:]
    signal_line = signal_line[-min_length_macd_signal:]
    
    # Calculate the MACD histogram
    macd_histogram = macd - signal_line
    
    return np.hstack([macd, signal_line, macd_histogram])


# Function to calculate Bollinger Bands
def calculate_bollinger_bands(prices, window_size=20, num_std_dev=2):
    moving_avg = calculate_moving_average(prices, window_size)
    rolling_std_dev = np.std(prices[-window_size:])
    upper_band = moving_avg + (rolling_std_dev * num_std_dev)
    lower_band = moving_avg - (rolling_std_dev * num_std_dev)
    return np.hstack([upper_band, lower_band])

# Function to calculate VWAP
def calculate_vwap(prices, volumes):
    vwap = np.sum(prices * volumes) / np.sum(volumes)
    return np.array([vwap])

# Function to calculate Volume Oscillator
def calculate_volume_oscillator(volumes, short_window=5, long_window=10):
    short_term_ma = np.mean(volumes[-short_window:])
    long_term_ma = np.mean(volumes[-long_window:])
    volume_oscillator = ((short_term_ma - long_term_ma) / long_term_ma) * 100
    return np.array([volume_oscillator])

# Function to process a single chunk
def process_chunk(chunk):
    open_prices = chunk[:, 0]
    high_prices = chunk[:, 1]
    low_prices = chunk[:, 2]
    close_prices = chunk[:, 3]
    adj_close_prices = chunk[:, 4]
    volumes = chunk[:, 5]
    
    open_stats = calculate_statistics(open_prices)
    high_stats = calculate_statistics(high_prices)
    low_stats = calculate_statistics(low_prices)
    close_stats = calculate_statistics(close_prices)
    adj_close_stats = calculate_statistics(adj_close_prices)
    volume_stats = calculate_statistics(volumes)
    
    price_movement = calculate_price_movement(open_prices, close_prices)
    
    moving_avg_close = calculate_moving_average(close_prices, 5)
    moving_avg_volume = calculate_moving_average(volumes, 5)
    
    rsi = calculate_rsi(close_prices, window_size=10)
    bollinger_bands = calculate_bollinger_bands(close_prices, window_size=10, num_std_dev=2)
    macd = calculate_macd(close_prices)
    
    vwap = calculate_vwap(close_prices, volumes)
    volume_oscillator = calculate_volume_oscillator(volumes, short_window=5, long_window=10)
    
    all_statistics = np.hstack([
        open_stats,
        high_stats,
        low_stats,
        close_stats,
        adj_close_stats,
        volume_stats,
        price_movement,
        moving_avg_close.flatten(),
        moving_avg_volume.flatten(),
        rsi,
        bollinger_bands.flatten(),
        macd.flatten(),
        vwap,
        volume_oscillator
    ])
    
    return all_statistics

# Main function
def main():
    # Read the CSV file
    df = pd.read_csv('AAPL.csv')

    num_chunks = len(df) // 10

    date_column = df.columns[0]
    dates = df[date_column]
    df = df.drop(df.columns[0], axis=1)

    all_statistics_arrays = []
    volatilities = []
    date_ranges = []

    for i in range(num_chunks - 1):
        chunk = df.iloc[i*10:(i+1)*10].values
        next_chunk = df.iloc[(i+1)*10:(i+2)*10].values

        statistics_vector = process_chunk(chunk)
        all_statistics_arrays.append(statistics_vector)
        
        high_prices = next_chunk[:, 1]
        low_prices = next_chunk[:, 2]
        volatility = calculate_volatility(high_prices, low_prices)
        volatilities.append(volatility)
        
        start_date = dates.iloc[i*10]
        end_date = dates.iloc[(i+1)*10 - 1]
        date_ranges.append(f"{start_date} to {end_date}")

    all_statistics_matrix = np.array(all_statistics_arrays)
    volatility_vector = np.array(volatilities)

    feature_vectors = []
    for i in range(len(all_statistics_matrix)):
        features = all_statistics_matrix[i]
        label = volatility_vector[i]
        feature_vector = np.hstack([features, label])
        feature_vectors.append(feature_vector)

    feature_matrix = np.array(feature_vectors)

    X = feature_matrix[:, :-1]
    y = feature_matrix[:, -1]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
        X_scaled, y, date_ranges, test_size=0.2, random_state=42
    )

    # Models to evaluate
    models = {
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(),
        'LightGBM': LGBMRegressor(),
        'LinearRegression': LinearRegression()
    }

    # Define hyperparameter grids for XGBoost and LightGBM
    param_grids = {
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'feature_fraction': [0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5],  # L1 regularization
            'reg_lambda': [0, 0.1, 0.5]  # L2 regularization
        }
    }

    # Store the results for predicted vs. actual volatilities
    results = []

    # Loop through each model, train it, and make predictions
    for name, model in models.items():
        print(f"Running model: {name}")

        # If the model has hyperparameters to tune, use RandomizedSearchCV
        if name in param_grids:
            random_search = RandomizedSearchCV(model, param_grids[name], n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            print(f"Best hyperparameters for {name}: {random_search.best_params_}")
        else:
            # If no hyperparameters to tune, fit the model directly
            best_model = model.fit(X_train, y_train)

        # Make predictions using the trained model
        y_pred = best_model.predict(X_test)

        # Model Evaluation
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"{name} Model Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R2: {r2:.4f}")
        print("-" * 40)

        # Append predicted vs actual along with date range and model name
        for actual, predicted, date_range in zip(y_test, y_pred, date_test):
            results.append([date_range, actual, predicted, name])

    # Create a DataFrame from the results and save to CSV
    results_df = pd.DataFrame(results, columns=["Date Range", "Actual Volatility", "Predicted Volatility", "Model"])
    results_df.to_csv("predicted_vs_actual_volatility_optimized.csv", index=False)

    print("Predicted vs Actual volatilities saved to 'predicted_vs_actual_volatility_optimized.csv'.")

main()
