import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to prepare sequences of stock data
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

# Prepare data for LSTM
def prepare_lstm_data(df, seq_length):
    # Assuming df contains a column with close prices
    close_prices = df['Close'].values

    # Calculate log returns
    log_returns = np.diff(np.log(close_prices))
    
    # Prepare sequences and labels
    sequences, labels = create_sequences(log_returns, seq_length)
    
    return sequences, labels

# LSTM Model Definition
class LSTMVolatilityModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMVolatilityModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Final output layer for volatility prediction
        
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        # Pass the final hidden state to the fully connected layer
        out = self.fc(out[:, -1, :])  # Take the last output of the sequence
        return out

# Training function for LSTM model
def train_lstm_model(model, train_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()  # Mean squared error loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for sequences, labels in train_loader:
            sequences = sequences.float().to(model.device).unsqueeze(-1)  # Ensure 3D input
            labels = labels.float().to(model.device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation function for LSTM model
def evaluate_lstm_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.float().to(model.device).unsqueeze(-1)  # Ensure 3D input
            outputs = model(sequences)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    # Calculate performance statistics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100  # MAPE

    # Print the statistics
    print(f"Model Performance Statistics:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"  R-squared (R²): {r2:.4f}")

    return predictions, actuals

# Main function
def main():
    # Read the CSV file
    df = pd.read_csv('AAPL.csv')

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Prepare training data
    seq_length = 10
    train_sequences, train_labels = prepare_lstm_data(train_df, seq_length=seq_length)

    # Convert to PyTorch tensors and create DataLoader
    train_dataset = TensorDataset(torch.tensor(train_sequences), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the LSTM model
    input_size = 1  # Log returns are a single feature
    hidden_size = 50  # Number of LSTM units
    num_layers = 2  # Number of LSTM layers

    model = LSTMVolatilityModel(input_size, hidden_size, num_layers)
    model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(model.device)

    # Train the model
    train_lstm_model(model, train_loader, num_epochs=100)

    # Prepare test data
    test_sequences, test_labels = prepare_lstm_data(test_df, seq_length=seq_length)

    # Convert to PyTorch tensors and create DataLoader
    test_dataset = TensorDataset(torch.tensor(test_sequences), torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model and display statistics
    predictions, actuals = evaluate_lstm_model(model, test_loader)

main()
