import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

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

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

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
        volatility = calculate_volatility(high_prices)
        volatilities.append(volatility)
        
        start_date = dates.iloc[i*10]
        end_date = dates.iloc[(i+1)*10 - 1]
        date_ranges.append(f"{start_date} to {end_date}")

    all_statistics_matrix = np.array(all_statistics_arrays)
    volatility_vector = np.array(volatilities)

    # Prepare features and labels
    X = all_statistics_matrix
    y = volatility_vector

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape the input to fit the LSTM model (batch_size, time_steps, input_size)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
        X_scaled, y, date_ranges, test_size=0.2, random_state=42
    )

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Define model parameters
    input_size = X_train.shape[2]
    hidden_size = 50
    num_layers = 2
    output_size = 1
    learning_rate = 0.001
    num_epochs = 100

    # Initialize the model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs.squeeze(), y_test)
        print(f'Test Loss: {test_loss.item():.4f}')

        # Calculate additional metrics if needed, like R^2 score
        y_pred = test_outputs.cpu().numpy()
        r2 = r2_score(y_test.cpu().numpy(), y_pred)
        print(f'R2 Score: {r2:.4f}')

if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main()
