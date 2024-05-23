import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from IPython.display import Image
import os
import numpy as np
import io
from PIL import Image
from torch.optim.lr_scheduler import StepLR

# Load and process the DataFrame
df = pd.read_pickle('sorted.pkl')  # Ensure this is already sorted by time
df['Temperature']  = df['Temperature'] + 273
# Convert the variables to PyTorch tensors
x = torch.tensor(df[['Temperature', 'Pressure', 'Relative Humidity', 'DNI', 'Wind Speed']].values, dtype=torch.float32)

import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(0)
        pe = torch.zeros(seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

class TransformerModel(nn.Module):
    def __init__(self, n_features, n_hiddens, n_layers, n_heads, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(5, n_hiddens)
        encoder_layers = nn.TransformerEncoderLayer(n_hiddens, n_heads, n_hiddens, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(n_hiddens, 4)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])
        return output

# Parameters
n_features = 32
n_hiddens = 128
n_layers = 128
n_heads = 128
n_epochs = 100
batch_size = 64
learning_rate = 0.0001

# Create the model
model = TransformerModel(n_features, n_hiddens, n_layers, n_heads)

# Loss function and optimizer
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
if not os.path.exists('plots'):
    os.makedirs('plots')

loss_values = []

fig, ax = plt.subplots(figsize=(10, 5))

for epoch in range(n_epochs):
    for i in range(0, len(x), batch_size):
        batch = x[i:i+batch_size]

        # Forward pass
        outputs = model(batch[:-1])

        # Concatenate the temperature from the previous step with the estimated variables
        outputs = torch.cat((batch[-1, 0].unsqueeze(0), outputs), dim=0)

        loss = criterion(outputs[1:], batch[-1, 1:])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 300 == 0:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # Store the loss values
        loss_values.append(loss.item())

        # Clear the plot
        ax.clear()

        # Plot the losses
        ax.plot(loss_values, label='Loss')

        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()

        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Open the image from the buffer
        img = Image.open(buf)

        # Display the plot
        clear_output(wait=True)
        display(img)

        # Save the plot to a file
        plt.savefig(f'plots/loss_plot_epoch_{epoch+1}.png')

    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")