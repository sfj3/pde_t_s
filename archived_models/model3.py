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
import torch.nn.functional as F
# Load and process the DataFrame
df = pd.read_pickle('sorted.pkl')  # Ensure this is already sorted by time

# Convert 'Temperature' and 'Pressure' to PyTorch tensors
x_sym = torch.tensor(df[['Temperature', 'Pressure']].values, dtype=torch.float32)
x_add = torch.tensor(df[['Relative Humidity', 'DNI', 'Wind Speed']].values, dtype=torch.float32)

import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch

def create_spline(params, num_outputs):
    num_inputs = params.shape[0]

    # Extract x and y coordinates from the parameter tensor
    x = torch.arange(num_inputs)
    y = params

    # Generate evenly spaced x-coordinates for the output
    x_output = torch.linspace(0, num_inputs - 1, num_outputs)

    # Calculate distances between each x-coordinate and the input x-coordinates
    distances = torch.abs(x_output.unsqueeze(1) - x.unsqueeze(0))

    # Find the indices of the four nearest points for each x-coordinate
    _, indices = torch.topk(distances, 4, largest=False, dim=1)

    # Extract the nearest x and y coordinates
    nearest_x = x[indices]
    nearest_y = y[indices]

    # Calculate the weights for interpolation
    weights = 1 / (distances.gather(1, indices) + 1e-8)
    normalized_weights = weights / weights.sum(dim=1, keepdim=True)

    # Perform cubic interpolation
    t = (x_output.view(-1, 1) - nearest_x[:, 0].view(-1, 1)) / (nearest_x[:, -1].view(-1, 1) - nearest_x[:, 0].view(-1, 1) + 1e-8)
    t_squared = t ** 2
    t_cubed = t ** 3

    coefficients = torch.tensor([
        [1, -3, 3, -1],
        [0, 3, -6, 3],
        [0, 0, 3, -3],
        [0, 0, 0, 1]
    ], dtype=torch.float32)

    basis_functions = torch.stack([
        t_cubed, t_squared, t, torch.ones_like(t)
    ], dim=-1)

    # Evaluate points on the multi-spline and sum along dimension 1
    output = (basis_functions @ coefficients @ nearest_y.unsqueeze(-1)).squeeze(-1)

    return output
    
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
    def __init__(self, n_features, n_hiddens, n_layers, n_heads, dropout=0.1, context_size=128):
        super(TransformerModel, self).__init__()
        self.context_size = context_size
        self.embedding_sym = nn.Linear(2, n_hiddens // 2)
        self.embedding_add = nn.Linear(3, n_hiddens // 2)
        self.pos_encoder = nn.Embedding(1000, n_hiddens)  # Learned positional embeddings
        encoder_layers = nn.TransformerEncoderLayer(n_hiddens, n_heads, n_hiddens, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.conv1 = nn.Conv1d(n_hiddens, n_hiddens, kernel_size=3, padding=1)  # Convolutional layer
        self.additive_attention = nn.Linear(n_hiddens, n_hiddens)  # Additive attention
        self.decoder = nn.Linear(n_hiddens, 2)
        self.nn_ABCD = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, src_sym, src_add):
        # Increase context size
        src_sym = src_sym[-self.context_size:]
        src_add = src_add[-self.context_size:]

        src_sym = self.embedding_sym(src_sym)
        src_add = self.embedding_add(src_add)
        src = torch.cat((src_sym, src_add), dim=-1)
        positions = torch.arange(0, src.size(0), dtype=torch.long).unsqueeze(1)
        src = src + self.pos_encoder(positions)
        output = self.transformer_encoder(src)

        # Residual connection
        residual = output

        output = output.permute(1, 2, 0)  # Reshape for convolutional layer
        output = self.conv1(output)
        output = output.permute(2, 0, 1)  # Reshape back

        # Additive attention
        attention_scores = torch.softmax(self.additive_attention(output), dim=0)
        output = torch.sum(attention_scores * output, dim=0)

        # Residual connection
        output = output + residual[-1]

        output = 100 * self.decoder(output)

        # Reshape output to (batch_size, sequence_length, input_size)
        output = output.unsqueeze(1)  # Add a dimension for sequence_length

        # Pass a dummy input through nn_ABCD to get ABCD values
        dummy_input = torch.ones(1, 1)  # Shape: (batch_size, input_size)
        ABCD = self.nn_ABCD(dummy_input)

        A, B, C, D = ABCD[:, 0], ABCD[:, 1], ABCD[:, 2], ABCD[:, 3]
        return output.squeeze(1), A, B, C, D

# Parameters
n_features = 128
n_hiddens = 16
n_layers = 16
n_heads = 16
n_epochs = 100
batch_size = 128
learning_rate = 0.01

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
        context_size = model.context_size

        # Forward pass
        outputs, A, B, C, D = model(batch_sym[:-1], batch_add[:-1])
        outputs[:,0],outputs[:,1] = create_spline(outputs[:,0],127).squeeze(),create_spline(outputs[:,1],127).squeeze()
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

        # Apply Softmax activation to the model's outputs
        outputs_softmax = F.softmax(outputs, dim=1)

        # Calculate the Cross-Entropy Loss
        loss_s = F.cross_entropy(outputs_softmax, batch_sym[-context_size+1:]) + torch.log(criterion(outputs, batch_sym[-context_size+1:]))

        # Calculate loss_entropy_pde_csts
        loss_entropy_pde_csts = 10 * torch.log(10 * torch.abs(rhs_output - lhs_output) * torch.abs(rhs_actual - lhs_actual)) + 1
        
        loss = loss_entropy_pde_csts.mean().item() * loss_s if not torch.isnan(loss_entropy_pde_csts.mean()) else loss_s

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.4)
        optimizer.step()
        # Store the loss values
        loss_s_values.append(loss_s.item())
        loss_entropy_pde_csts_values.append(loss_entropy_pde_csts.mean().item())
        total_loss_values.append(loss.item())
        print("LHS Output:", lhs_output.mean().item(), "RHS Output:", rhs_output.mean().item())
        print("LHS Actual:", lhs_actual.mean().item(), "RHS Actual:", rhs_actual.mean().item())
        print("A:", A, "B:", B, "C:", C, "D:", D)
        print("Outputs are ", outputs, "Actual is ", batch_sym[1:], 'inputs are: ', batch_sym[:-1])
        print("Entropy PDE Loss:", loss_entropy_pde_csts.mean().item())
        print("transformer Loss is ", loss_s)
        print('total loss is ', loss)

        # Clear the plot
        ax.clear()

        # Plot the losses
        ax.plot(loss_s_values, label='Transformer Loss')
        ax.plot(loss_entropy_pde_csts_values, label='Entropy PDE Loss')
        ax.plot(np.sqrt(total_loss_values), label='Geometric Mean Loss')


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