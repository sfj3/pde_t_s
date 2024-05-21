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
num_encoder_layers = 8
num_decoder_layers = 8
dim_feedforward = 64
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

for epoch in range(num_epochs):
    for start in range(0, x_sym_tensor.size(0) - window_size, window_size):
        end = start + window_size
        seq_input = x_sym_tensor[start:end, :]
        
        # Create batches within the window
        for i in range(0, seq_input.size(0) - batch_size, batch_size):
            batch_input = seq_input[i:i+batch_size, :]
            batch_target = seq_input[i+1:i+batch_size+1, :]  # Target is the next step

            # Forward pass
            output = model(batch_input, batch_target)

            # Apply the transformation to tensors
            int_output = tensor_to_int(output)
            int_batch_input = tensor_to_int(batch_input)
            int_batch_target = tensor_to_int(batch_target)

            # Normalize outputs
            output_df = torch.stack(((int_output % 150) - 90, torch.div(int_output, 150).float() + 800), dim=-1)
            input_df = torch.stack(((int_batch_input % 150) - 90, torch.div(int_batch_input, 150).float() + 800), dim=-1)
            target_df = torch.stack(((int_batch_target % 150) - 90, torch.div(int_batch_target, 150).float() + 800), dim=-1)

            lhs_list, rhs_list = [], []

            for step in range(4):
                Radiation = x_add[0, start+i+step+1, 1]  # Use the next step for target
                Wind_Speed = x_add[0, start+i+step+1, 2]  # Use the next step for target
                humidity_change = x_add[0, start+i+step+1, 0] - x_add[0, start+i+step, 0]  # Use the next step for target

                lhs = torch.log((1 / (target_df[:, step, 0] + 273) * (target_df[:, step, 1] / 850) ** 8.314 + (850 / target_df[:, step, 1]) ** 8.314)) - torch.log((1 / (input_df[:, step, 0] + 273) * (input_df[:, step, 1] / 850) ** 8.314 + (850 / input_df[:, step, 1]) ** 8.314))
                rhs = (1 / (target_df[:, step, 0] + 273)) * (Radiation + humidity_change + 1) + Wind_Speed * (torch.log((1 / (target_df[:, step, 0] + 273) * (target_df[:, step, 1] / 850) ** 8.314 + (850 / target_df[:, step, 1]) ** 8.314)))

                lhs_list.append(lhs)
                rhs_list.append(rhs)

            lhs_tensor = torch.stack(lhs_list, dim=1)
            rhs_tensor = torch.stack(rhs_list, dim=1)

            # Solve for A, B, C, D
            solution = torch.linalg.lstsq(rhs_tensor, lhs_tensor).solution.reshape(batch_size, 4)
            A, B, C, D = solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]

            # Recalculate RHS using the obtained A, B, C, D
            Radiation = x_add[0, end, 1]  # Use the last step of the window
            Wind_Speed = x_add[0, end, 2]  # Use the last step of the window
            humidity_change = x_add[0, end, 0] - x_add[0, end-1, 0]  # Use the last step of the window
            rhs_output = (1 / (output_df[:, -1, 0] + 273)) * (A * Radiation + B * humidity_change + C) + D * Wind_Speed * (torch.log((1 / (output_df[:, -1, 0] + 273) * (output_df[:, -1, 1] / 850) ** 8.314 + (850 / output_df[:, -1, 1]) ** 8.314)))

            # Calculate the losses
            loss_entropy_pde_csts = 10 * torch.abs(lhs_tensor[:, -1] - rhs_output)
            loss_actual = nn.CrossEntropyLoss()(torch.log(output_df[:, :, 1] / (output_df[:, :, 0] + 273)), torch.log(target_df[:, :, 1] / (target_df[:, :, 0] + 273)))

            # Compute the binary cross-entropy loss
            bce_loss = nn.BCELoss()(output, batch_target)
            
            # Combine the losses
            loss = torch.max(torch.stack([bce_loss, loss_entropy_pde_csts.mean(), loss_actual]))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()

            # Store loss values for plotting
            bce_loss_values.append(bce_loss.item())
            loss_entropy_pde_csts_values.append(loss_entropy_pde_csts.mean().item())
            loss_actual_values.append(loss_actual.item())
            total_loss_values.append(loss.item())

        # Print the loss for every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Plot the losses
        clear_output(wait=True)
        plt.figure(figsize=(10, 5))
        plt.plot(bce_loss_values, label='BCE Loss')
        plt.plot(loss_entropy_pde_csts_values, label='Entropy PDE Consts Loss')
        plt.plot(loss_actual_values, label='Actual Loss')
        plt.plot(total_loss_values, label='Total Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('plots/loss_plot.png')
        plt.close()

        # Display the plot
        display(Image(filename='plots/loss_plot.png'))

# Convert the binary output to binary strings
output_binary = output.int().apply_(lambda x: ''.join(map(str, x.tolist()))).tolist()

# Convert the binary strings back to temperature and pressure values
temp_pressure_output = pd.Series(output_binary).apply(lambda binary_str: pd.Series(((int(binary_str, 2) % 150) - 90, (int(binary_str, 2) // 150) + 800)))