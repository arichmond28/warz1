import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
      # Ensure x is 3D: (batch_size, sequence_length, input_size)
      if len(x.shape) == 2:
          x = x.unsqueeze(0)  # Add batch dimension if necessary
    
      # Initialize hidden and cell states
      h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
      c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
    
      # Forward propagate LSTM
      out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
    
      # Take the last output of the LSTM for each sequence
      out = out[:, -1, :]  # out shape is now [batch_size, hidden_size]
      out = self.fc(out)  # out shape is now [batch_size, 1]
    
      return out

# Function to prepare sequences and volatility labels for training
def prepare_lstm_data(df, seq_length=10):
    close_prices = df['Close'].values
    
    # Calculate log returns
    log_returns = np.diff(np.log(close_prices))
    
    sequences = []
    labels = []
    
    for i in range(len(log_returns) - seq_length - 10):  # Ensure we have enough future days to compute volatility
        # Get the sequence of log returns for the past `seq_length` days
        seq = log_returns[i:i + seq_length]
        sequences.append(seq)
        
        # Calculate the volatility (standard deviation of log returns) for the next 10 days
        future_log_returns = log_returns[i + seq_length:i + seq_length + 10]
        future_volatility = np.std(future_log_returns)
        labels.append(future_volatility)
    
    return np.array(sequences), np.array(labels)

# Main function
def main():
    # Load the data
    df = pd.read_csv('AAPL.csv')
    
    # Prepare sequences and labels
    seq_length = 10
    sequences, labels = prepare_lstm_data(df, seq_length)
    
    # Scale the data
    scaler = StandardScaler()
    sequences_scaled = scaler.fit_transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape)
    
    # Convert to PyTorch tensors
    sequences_tensor = torch.tensor(sequences_scaled, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)  # Reshape labels to [batch_size, 1]
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(sequences_tensor, labels_tensor)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the LSTM model
    input_size = sequences.shape[-1]  # Number of features per timestep
    hidden_size = 50
    num_layers = 2
    output_size = 1  # Predicting one output: the future volatility
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for sequences_batch, labels_batch in train_loader:
            sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
            
            # Forward pass
            outputs = model(sequences_batch)  # Outputs have shape [batch_size, 1]
            
            # Compute the loss
            loss = criterion(outputs, labels_batch)  # Ensure both outputs and labels are [batch_size, 1]
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    
    # Evaluate the model on validation data
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for sequences_batch, labels_batch in val_loader:
            sequences_batch, labels_batch = sequences_batch.to(device), labels_batch.to(device)
            outputs = model(sequences_batch)  # Outputs have shape [batch_size, 1]
            loss = criterion(outputs, labels_batch)  # Ensure both outputs and labels are [batch_size, 1]
            val_loss += loss.item()
        
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')

    # Save the model
    torch.save(model.state_dict(), 'lstm_volatility_model.pth')
    print("Model saved to 'lstm_volatility_model.pth'.")

main()
