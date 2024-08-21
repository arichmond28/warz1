import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold

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
    # Calculate log returns
    log_returns = np.diff(np.log(prices))
    
    # Calculate standard deviation of log returns
    volatility = np.std(log_returns)
    
    # Annualize the volatility if required
    if annualized:
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
    # Split the chunk into respective features
    open_prices = chunk[:, 0]
    high_prices = chunk[:, 1]
    low_prices = chunk[:, 2]
    close_prices = chunk[:, 3]
    adj_close_prices = chunk[:, 4]
    volumes = chunk[:, 5]
    
    # Calculate stats for each feature
    open_stats = calculate_statistics(open_prices)
    high_stats = calculate_statistics(high_prices)
    low_stats = calculate_statistics(low_prices)
    close_stats = calculate_statistics(close_prices)
    adj_close_stats = calculate_statistics(adj_close_prices)
    volume_stats = calculate_statistics(volumes)
    
    # Calculate price movements
    price_movement = calculate_price_movement(open_prices, close_prices)
    
    # Calculate moving averages (e.g., 5-day)
    moving_avg_close = calculate_moving_average(close_prices, 5)
    moving_avg_volume = calculate_moving_average(volumes, 5)
    
    # Calculate technical indicators
    rsi = calculate_rsi(close_prices, window_size=10)
    bollinger_bands = calculate_bollinger_bands(close_prices, window_size=10, num_std_dev=2)
    vwap = calculate_vwap(close_prices, volumes)
    
    # Volume oscillator
    volume_oscillator = calculate_volume_oscillator(volumes, short_window=5, long_window=10)
    
    # Concatenate all statistics into a single vector
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
        vwap,
        volume_oscillator
    ])
    
    return all_statistics

# Main function
def main():
    # Read the CSV file
    df = pd.read_csv('AAPL.csv')

    # Determine the number of chunks
    num_chunks = len(df) // 10

    # Drop the first column if it's not needed (like a date column)
    date_column = df.columns[0]
    dates = df[date_column]
    df = df.drop(df.columns[0], axis=1)

    # Initialize lists to store calculated data
    all_statistics_arrays = []
    volatilities = []
    date_ranges = []

    # Precompute volatilities for the entire dataset
    for i in range(num_chunks - 1):  # Exclude the last chunk since there's no subsequent period
        # Slice the dataframe to get 10 rows and convert them to a numpy array
        chunk = df.iloc[i*10:(i+1)*10].values
        next_chunk = df.iloc[(i+1)*10:(i+2)*10].values

        # Process the chunk and calculate all statistics except volatility
        statistics_vector = process_chunk(chunk)
        all_statistics_arrays.append(statistics_vector)
        
        # Calculate and store volatility for the subsequent 10-day period
        high_prices = next_chunk[:, 1]
        low_prices = next_chunk[:, 2]
        volatility = calculate_volatility(high_prices, low_prices)
        volatilities.append(volatility)
        
        # Store the date range for this chunk (corresponding to the period used to calculate features)
        start_date = dates.iloc[i*10]
        end_date = dates.iloc[(i+1)*10 - 1]
        date_ranges.append(f"{start_date} to {end_date}")

    # Convert lists to numpy arrays
    all_statistics_matrix = np.array(all_statistics_arrays)
    volatility_vector = np.array(volatilities)

    # Initialize a list for the final feature vectors (features + label)
    feature_vectors = []

    # Build feature vectors with [features... label]
    for i in range(len(all_statistics_matrix)):
        features = all_statistics_matrix[i]
        label = volatility_vector[i]
        feature_vector = np.hstack([features, label])
        feature_vectors.append(feature_vector)

    # Convert the list of feature vectors to a numpy array
    feature_matrix = np.array(feature_vectors)

    # Split feature matrix into features (X) and labels (y)
    X = feature_matrix[:, :-1]  # All columns except the last (features)
    y = feature_matrix[:, -1]   # The last column (label)

    # Split data into training and testing sets (80% train, 20% test), with a fixed random state
    X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
        X, y, date_ranges, test_size=0.2, random_state=42
    )

    # Models to evaluate
    models = {
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'Bagging': BaggingRegressor(),
        'ExtraTrees': ExtraTreesRegressor(),
        'LinearRegression': LinearRegression()
    }

    # Define hyperparameter grids for each model (you can expand this as needed)
    param_grids = {
        'DecisionTree': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]},
        'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
        # Add more parameter grids as needed for each model
    }

    # Store the results for predicted vs. actual volatilities
    results = []

    # Loop through each model, train it, and make predictions
    for name, model in models.items():
        print(f"Running model: {name}")

        # If the model has hyperparameters to tune, use GridSearchCV
        if name in param_grids:
            grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            print(f"Best hyperparameters for {name}: {grid_search.best_params_}")
        else:
            # If no hyperparameters to tune (e.g., LinearRegression), just fit the model
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




