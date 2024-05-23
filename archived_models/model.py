import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from IPython.display import Image
import os
import numpy as np
from PIL import Image
import io
# Load and process the DataFrame
df = pd.read_pickle('sorted.pkl')  # Ensure this is already sorted by time
# Assuming df is your DataFrame
x_sym = torch.tensor(np.log(df['Temperature'].to_numpy() * (850 / df['Pressure'].to_numpy()) ** 8.315), dtype=torch.float32)

# Calculate the padding required
pad_size = (16 - x_sym.size(0) % 16) % 16

# Pad the tensor
if pad_size > 0:
    x_sym = torch.nn.functional.pad(x_sym, (0, pad_size))

# Reshape to have inner dimension of 16
x_sym_tensor = x_sym.view(-1, 16)
x_add = torch.tensor(df[['Relative Humidity', 'DNI', 'Wind Speed']].values, dtype=torch.float32)
x_add = x_add.unsqueeze(0)

class BinaryTransformerWithLSTM(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=True, norm_first=False):
        super(BinaryTransformerWithLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=d_model, num_layers=3, batch_first=batch_first)
        
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
        
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.fc3 = nn.Linear(d_model, d_model)
        self.fc4 = nn.Linear(d_model, d_model)
        self.fc5 = nn.Linear(d_model, d_model)
        self.fc6 = nn.Linear(d_model, d_model)
        self.fc7 = nn.Linear(d_model, d_model)
        self.fc8 = nn.Linear(d_model, d_model)
        self.fc9 = nn.Linear(d_model, d_model)
        self.fc10 = nn.Linear(d_model, d_model)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, src, tgt):
        src = self.conv1(src.transpose(1, 2))
        src = self.conv2(src)
        src = self.batch_norm1(src)
        src = self.dropout1(src)
        
        src = src.transpose(1, 2)
        
        lstm_out, _ = self.lstm(src)
        
        output = self.transformer(lstm_out, tgt)
        
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.fc4(output)
        output = self.fc5(output)
        output = self.fc6(output)
        output = self.fc7(output)
        output = self.fc8(output)
        output = self.fc9(output)
        output = self.fc10(output)
        
        output = self.sigmoid(output)
        
        return output

# Function to convert tensor to int representation
def tensor_to_int(t):
    return (t * 2**torch.arange(t.size(-1) - 1, -1, -1, device=t.device)).sum(dim=-1)

# Initialize the model with flexible parameters
def initialize_model(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-5, batch_first=False, norm_first=False):
    return BinaryTransformerWithLSTM(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

# Create the optimizer with flexible parameters
def initialize_optimizer(model, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
    return optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

# Example usage
d_model = 16
nhead = 2
num_encoder_layers = 32
num_decoder_layers = 32
dim_feedforward = 128
dropout = 0.01
activation = 'relu'
layer_norm_eps = 1e-5
batch_first = True
norm_first = False
learning_rate = 5  # Start with a higher learning rate
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0

model = initialize_model(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first)
optimizer = initialize_optimizer(model, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)

# Training loop with sliding window approach
num_epochs = 100
window_size = 1000  # Set window size for each training step
batch_size = 32  # Set an appropriate batch size

# Lists to store loss values for plotting
bce_loss_values = []
loss_entropy_pde_csts_values = []
loss_actual_values = []
total_loss_values = []

# Directory to save the plots
if not os.path.exists('plots'):
    os.makedirs('plots')
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
        print("Outputs are ", outputs, "Actual is ", batch_sym[1:], 'inputs are: ', batch_sym[:-1])

        # Calculate loss_entropy_pde_csts
        loss_entropy_pde_csts = 100 * torch.abs(rhs_output - lhs_actual) * torch.abs(rhs_output - rhs_actual) * torch.abs(lhs_output - rhs_actual) * torch.abs(lhs_output - lhs_actual)
        print("Entropy PDE Loss:", loss_entropy_pde_csts.mean().item())
        print("transformer Loss is ", loss_s)

        loss = loss_entropy_pde_csts.mean().item() + loss_s if not torch.isnan(loss_entropy_pde_csts.mean()) else loss_s
        print('total loss is ', loss)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
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

    # Set y-scale to logarithmic
    ax.set_yscale('log')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss (log scale)')
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

    print(f"Epoch [{epoch+1}/{n_epochs}] completed.")

# Convert the binary output to binary strings
output_binary = output.int().apply_(lambda x: ''.join(map(str, x.tolist()))).tolist()

# Convert the binary strings back to temperature and pressure values
temp_pressure_output = pd.Series(output_binary).apply(lambda binary_str: pd.Series(((int(binary_str, 2) % 150) - 90, (int(binary_str, 2) // 150) + 800)))