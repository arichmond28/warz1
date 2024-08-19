import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

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

# Function to calculate volatility
def calculate_volatility(high_prices, low_prices):
    daily_range_volatility = np.mean(high_prices - low_prices)
    return daily_range_volatility

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
    df = df.drop(df.columns[0], axis=1)

    # Initialize an empty list to store the numpy arrays
    numpy_arrays = []
    all_statistics_arrays = []
    volatilities = []

    # Loop over the range of chunks
    for i in range(num_chunks):
        # Slice the dataframe to get 10 rows and convert them to a numpy array
        chunk = df.iloc[i*10:(i+1)*10].values
        numpy_arrays.append(chunk)
        
        # Process the chunk and calculate all statistics except volatility
        statistics_vector = process_chunk(chunk)
        all_statistics_arrays.append(statistics_vector)
        
        # Calculate and store volatility separately
        high_prices = chunk[:, 1]
        low_prices = chunk[:, 2]
        volatility = calculate_volatility(high_prices, low_prices)
        volatilities.append(volatility)

    # Convert list of all statistics arrays and volatilities to final numpy arrays for further use
    all_statistics_matrix = np.array(all_statistics_arrays)
    volatility_vector = np.array(volatilities)

    # Display the shape of the final matrix and volatility vector, and the first few rows as a sample
    print("All Statistics Matrix Shape:", all_statistics_matrix.shape)
    print("Volatility Vector Shape:", volatility_vector.shape)
    print("Sample of Statistics Matrix (first 2 rows):\n", all_statistics_matrix[:2])
    print("Sample of Volatility Vector (first 2 values):\n", volatility_vector[:2])

main()

