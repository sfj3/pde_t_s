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

# Convert 'Temperature' and 'Pressure' to PyTorch tensors
x_sym = torch.tensor(df[['Temperature', 'Pressure']].values, dtype=torch.float32)
x_add = torch.tensor(df[['Relative Humidity', 'DNI', 'Wind Speed']].values, dtype=torch.float32)

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
        self.embedding_sym = nn.Linear(2, n_hiddens // 2)
        self.embedding_add = nn.Linear(3, n_hiddens // 2)
        self.pos_encoder = PositionalEncoding(n_hiddens)
        encoder_layers = nn.TransformerEncoderLayer(n_hiddens, n_heads, n_hiddens, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(n_hiddens, 2)
        self.nn_ABCD = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, src_sym, src_add):
        src_sym = self.embedding_sym(src_sym)
        src_add = self.embedding_add(src_add)
        src = torch.cat((src_sym, src_add), dim=-1)
        src = src + self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = 100 * self.decoder(output[-1])
        ABCD = self.nn_ABCD(output)
        A, B, C, D = ABCD[:, 0], ABCD[:, 1], ABCD[:, 2], ABCD[:, 3]
        return output, A, B, C, D

# Parameters
n_features = 8
n_hiddens = 128
n_layers = 64
n_heads = 32*2
n_epochs = 100
batch_size = 32*2
learning_rate = 0.0004

# Create the model
model = TransformerModel(n_features, n_hiddens, n_layers, n_heads)

# Loss function and optimizer
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
A_list, B_list, C_list, D_list = [], [], [], []
lhs_list, rhs_list = [], []

import os
if not os.path.exists('plots'):
    os.makedirs('plots')

loss_s_values = []
loss_entropy_pde_csts_values = []
total_loss_values = []

if not os.path.exists('plots'):
    os.makedirs('plots')

loss_s_values = []
loss_entropy_pde_csts_values = []
total_loss_values = []

fig, ax = plt.subplots(figsize=(10, 5))

for epoch in range(n_epochs):
    for i in range(0, len(x_sym), batch_size):
        batch_sym = x_sym[i:i+batch_size]
        batch_add = x_add[i:i+batch_size]

        # Forward pass
        outputs, A, B, C, D = model(batch_sym[:-1], batch_add[:-1])

        # Calculate LHS and RHS
        Radiation = batch_add[1:, 1]
        Wind_Speed = batch_add[1:, 2]
        humidity_change = batch_add[1:, 0] - batch_add[:-1, 0]

        lhs_output = torch.log((1 / (outputs[:, 0] + 273) * (outputs[:, 1] / 850) ** 8.314 + (850 / outputs[:, 1]) ** 8.314)) - torch.log((1 / (batch_sym[:-1, 0] + 273) * (batch_sym[:-1, 1] / 850) ** 8.314 + (850 / batch_sym[:-1, 1]) ** 8.314))
        rhs_output = (1 / (outputs[:, 0] + 273)) * (A * Radiation + B * humidity_change + C) + D * Wind_Speed * (torch.log((1 / (outputs[:, 0] + 273) * (outputs[:, 1] / 850) ** 8.314 + (850 / outputs[:, 1]) ** 8.314)))

        lhs_actual = torch.log((1 / (batch_sym[1:, 0] + 273) * (batch_sym[1:, 1] / 850) ** 8.314 + (850 / batch_sym[1:, 1]) ** 8.314)) - torch.log((1 / (batch_sym[:-1, 0] + 273) * (batch_sym[:-1, 1] / 850) ** 8.314 + (850 / batch_sym[:-1, 1]) ** 8.314))
        rhs_actual = (1 / (batch_sym[1:, 0] + 273)) * (Radiation + humidity_change + 1) + Wind_Speed * (torch.log((1 / (batch_sym[1:, 0] + 273) * (batch_sym[1:, 1] / 850) ** 8.314 + (850 / batch_sym[1:, 1]) ** 8.314)))

        lhs_list.append(lhs_output.mean().item())
        rhs_list.append(rhs_output.mean().item())

        print("LHS Output:", lhs_output.mean().item(), "RHS Output:", rhs_output.mean().item())
        print("LHS Actual:", lhs_actual.mean().item(), "RHS Actual:", rhs_actual.mean().item())
        print("A:", A, "B:", B, "C:", C, "D:", D)

        loss_s = torch.log((criterion(outputs, batch_sym[1:]) + torch.std(outputs - batch_sym[1:]) + torch.max(torch.abs(criterion(outputs[1:], batch_sym[2:]) - criterion(outputs[:-1], batch_sym[1:-1]))))**4)
        print("Outputs are ", outputs, "Actual is ", batch_sym[1:], 'inputs are: ', batch_sym[:-1])

        # Calculate loss_entropy_pde_csts
        loss_entropy_pde_csts = torch.log(2*torch.abs(rhs_output - lhs_output) * torch.abs(rhs_actual - lhs_actual))
        print("Entropy PDE Loss:", loss_entropy_pde_csts.mean().item())
        print("transformer Loss is ", loss_s)

        loss = loss_entropy_pde_csts.mean().item() + loss_s if not torch.isnan(loss_entropy_pde_csts.mean()) else loss_s
        print('total loss is ', loss)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.4)
        optimizer.step()

        # Store the loss values
        loss_s_values.append(loss_s.item())
        loss_entropy_pde_csts_values.append(loss_entropy_pde_csts.mean().item())
        total_loss_values.append(loss.item())

        # Clear the plot
        ax.clear()

        # Plot the losses
        ax.plot(loss_s_values, label='Transformer Loss')
        ax.plot(loss_entropy_pde_csts_values, label='Entropy PDE Loss')
        ax.plot(total_loss_values, label='Total Loss')


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