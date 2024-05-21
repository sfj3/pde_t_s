import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from IPython.display import Image
import os
import numpy as np

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
        self.A = nn.Parameter(torch.randn(1))
        self.B = nn.Parameter(torch.randn(1))
        self.C = nn.Parameter(torch.randn(1))
        self.D = nn.Parameter(torch.randn(1))

    def forward(self, src_sym, src_add):
        src_sym = self.embedding_sym(src_sym)
        src_add = self.embedding_add(src_add)
        src = torch.cat((src_sym, src_add), dim=-1)
        src = src + self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = 100 * self.decoder(output[-1])
        return output

# Parameters
n_features = 5
n_hiddens = 256
n_layers = 12
n_heads = 32
n_epochs = 100
batch_size = 32
learning_rate = 0.0001

# Create the model
model = TransformerModel(n_features, n_hiddens, n_layers, n_heads)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
A_list, B_list, C_list, D_list = [], [], [], []
lhs_list, rhs_list = [], []

for epoch in range(n_epochs):
    for i in range(0, len(x_sym), batch_size):
        batch_sym = x_sym[i:i+batch_size]
        batch_add = x_add[i:i+batch_size]

        # Forward pass
        outputs = model(batch_sym[:-1], batch_add[:-1])

        # Calculate LHS and RHS
        Radiation = batch_add[1:, 1]
        Wind_Speed = batch_add[1:, 2]
        humidity_change = batch_add[1:, 0] - batch_add[:-1, 0]

        lhs_output = torch.log((1 / (outputs[:, 0] + 273) * (outputs[:, 1] / 850) ** 8.314 + (850 / outputs[:, 1]) ** 8.314)) - torch.log((1 / (batch_sym[:-1, 0] + 273) * (batch_sym[:-1, 1] / 850) ** 8.314 + (850 / batch_sym[:-1, 1]) ** 8.314))
        rhs_output = (1 / (outputs[:, 0] + 273)) * (model.A * Radiation + model.B * humidity_change + model.C) + model.D * Wind_Speed * (torch.log((1 / (outputs[:, 0] + 273) * (outputs[:, 1] / 850) ** 8.314 + (850 / outputs[:, 1]) ** 8.314)))

        lhs_actual = torch.log((1 / (batch_sym[1:, 0] + 273) * (batch_sym[1:, 1] / 850) ** 8.314 + (850 / batch_sym[1:, 1]) ** 8.314)) - torch.log((1 / (batch_sym[:-1, 0] + 273) * (batch_sym[:-1, 1] / 850) ** 8.314 + (850 / batch_sym[:-1, 1]) ** 8.314))
        rhs_actual = (1 / (batch_sym[1:, 0] + 273)) * (Radiation + humidity_change + 1) + Wind_Speed * (torch.log((1 / (batch_sym[1:, 0] + 273) * (batch_sym[1:, 1] / 850) ** 8.314 + (850 / batch_sym[1:, 1]) ** 8.314)))

        lhs_list.append(lhs_output.mean().item())
        rhs_list.append(rhs_output.mean().item())
        
        print("LHS Output:", lhs_output.mean().item(), "RHS Output:", rhs_output.mean().item())
        print("LHS Actual:", lhs_actual.mean().item(), "RHS Actual:", rhs_actual.mean().item())
        print("A:", model.A, "B:", model.B, "C:", model.C, "D:", model.D)
        loss_s = criterion(outputs, batch_sym[1:])
        print( "Outputs are", outputs, "Actual is", batch_sym[1:])
        
        # Calculate loss_entropy_pde_csts
        loss_entropy_pde_csts = torch.abs(rhs_output - lhs_actual)*torch.abs(rhs_output - rhs_actual)*torch.abs(lhs_output - rhs_actual)*torch.abs(lhs_output - lhs_actual)
        print("Entropy PDE Loss:", loss_entropy_pde_csts.mean().item())
        print("transformer Loss is ", loss_s)
        loss = loss_entropy_pde_csts.mean().item() + loss_s if not torch.isnan(loss_entropy_pde_csts.mean()) else loss_s
        print('total loss is ',loss)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
    print("A:", model.A.item(), "B:", model.B.item(), "C:", model.C.item(), "D:", model.D.item())
    A_list.append(model.A.item())
    B_list.append(model.B.item())
    C_list.append(model.C.item())
    D_list.append(model.D.item())

# Plot the evolution of constants
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(A_list)
plt.title("Evolution of Constant A")

plt.subplot(2, 2, 2)
plt.plot(B_list)
plt.title("Evolution of Constant B")

plt.subplot(2, 2, 3)
plt.plot(C_list)
plt.title("Evolution of Constant C")

plt.subplot(2, 2, 4)
plt.plot(D_list)
plt.title("Evolution of Constant D")

plt.tight_layout()
plt.show()

# Plot LHS and RHS
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(lhs_list, label="LHS Output")
plt.plot(rhs_list, label="RHS Output")
plt.legend()
plt.title("LHS and RHS Output")

plt.subplot(1, 2, 2)
plt.plot(lhs_actual, label="LHS Actual")
plt.plot(rhs_actual, label="RHS Actual")
plt.legend()
plt.title("LHS and RHS Actual")

plt.tight_layout()
plt.show()