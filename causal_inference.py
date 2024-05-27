#scale the driver maps so that you can see color variation better?, ya i def need to debug the plotting

import torch
from torch import Tensor
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from IPython.display import Image
import io
from PIL import Image

# Load the data
df = pd.read_pickle('sorted.pkl')
df['Temperature'] = df['Temperature'] + 273
df['Relative Humidity'] = df['Relative Humidity']
df['DNI'] = df['DNI']
df["Wind Speed"] = df["Wind Speed"]
df['Pressure'] = df['Pressure']

# Calculate derivatives (differences between consecutive values)
df['Temperature_diff'] = df['Temperature'].diff()
df['Relative Humidity_diff'] = df['Relative Humidity'].diff()
df['DNI_diff'] = df['DNI'].diff()
df['Wind Speed_diff'] = df['Wind Speed'].diff()
df['Pressure_diff'] = df['Pressure'].diff()

# Fill NaN values with 0 (first row will have NaN for derivatives)
df.fillna(0, inplace=True)

x = torch.abs(torch.tensor(df[['Temperature', 'Pressure', 'Relative Humidity', 'DNI', 'Wind Speed',
                                'Temperature_diff', 'Pressure_diff', 'Relative Humidity_diff', 'DNI_diff', 'Wind Speed_diff']].values))

# Bias normalization with safety
feature_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Example indices
batch_norm = nn.BatchNorm1d(len(feature_indices), affine=False)
x_features = x[:, feature_indices]
x_features_normalized = torch.e + batch_norm((x_features.type('torch.FloatTensor') + 0.0000000000001) * 10.00000000000)
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.f_3 = nn.TransformerEncoderLayer(d_model=(24),nhead=24)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=1)
        # then dense
        self.dense = nn.Linear(20, 20)
        # then relu at the end, well whatever will bring them between 0 and 1, so maybe 



        #then for the pe 
        num_layers = 2
        self.num_layers = num_layers
        for _ in range(num_layers):
            self.scale_ = nn.Parameter(torch.ones(10))
            self.bias_ = nn.Parameter(torch.zeros(10))

        #then for the mu model
        for _ in range(2):
            self.scale_mu_ = nn.Parameter(torch.ones(2))
            self.bias_mu_ = nn.Parameter(torch.zeros(2))

    def forward(self, src,s_current_estimated):
        x_prev_24_with_fft = torch.abs(torch.cat((src,torch.fft.fft(src)))).view(10,2,24)
        x = torch.relu(x_prev_24_with_fft)
        x = self.f_3(x)
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.squeeze(1)
        x = x.view(24,20)
        x = self.dense(x)
        x = x[-1]
        x = torch.relu(x)
        # print(x.shape)
        return x
    

    def pe_forward(self, x: Tensor) -> Tensor:
        return torch.acos(torch.tanh(x))     
    def u1_forward(self, x: Tensor) -> Tensor: #this is the forward of the current entropy estimate model on just t and p 
        num_layers = self.num_layers
        for _ in range(num_layers):
            x = self.scale_mu_*x + self.bias_mu_
            x = torch.sigmoid(x)
        return x
        

    def u1_inverse(self, output: Tensor) -> Tensor:
        num_layers = self.num_layers
        x = output
        for _ in reversed(range(num_layers)):
            x = torch.log(x/ (1 - x))
            x = (x - self.bias_mu_)/self.scale_mu_
        return x
         
# Load the saved model weights
xfmr = TransformerModel()
xfmr.load_state_dict(torch.load('causal_weights.pth'))
xfmr.eval()  # Set the model to evaluation mode

# Initialize lists to store losses
entropy_losses = []
temp_pressure_losses = []
total_losses = []
inductive_losses = []

# Concatenate data for inference
x_features_normalized = torch.cat((x_features_normalized, x_features_normalized, x_features_normalized, x_features_normalized,
                                   x_features_normalized, x_features_normalized, x_features_normalized), dim=0)

# # Perform inference and evaluate losses
# with torch.no_grad():
#     for i in range(x_features_normalized.shape[0]):
#         if i == len(x_features_normalized) - 1:
#             continue  # Skip the last step

#         if i <= 25:
#             continue  # Skip the first 25 steps

#         entropy_t2 = torch.log(torch.e + x_features_normalized[i+1, 0]) - 8.314 * torch.log(torch.e + x_features_normalized[i+1, 1] / 2)
#         entropy_t1 = torch.log(torch.e + x_features_normalized[i, 0]) - 8.314 * torch.log(torch.e + x_features_normalized[i, 1] / 2)
#         d_s_actual = entropy_t2 - entropy_t1

#         x_prev_24 = x_features_normalized[i-24:i, ]

#         s_current_estimated = xfmr.u1_forward(x_prev_24[-2:, 0:2])
#         x_prev_24_pe = xfmr.pe_forward(x_prev_24)
#         out = xfmr(x_prev_24_pe, s_current_estimated)

#         x_current = x_features_normalized[i]
#         t_p_estimate = torch.abs(torch.sum(out.view(2, 10) * x_current, axis=-1))
#         d_s_predicted = t_p_estimate[-1]
#         s_new_estimate = d_s_predicted
#         t_predicted = s_new_estimate + 8.314 * torch.log(torch.e + x_features_normalized[i, 1] / 2)
#         p_predicted = s_new_estimate - torch.log(torch.e + x_features_normalized[i, 0])

#         s_current_entropy_form = s_current_estimated[0] - 8.314 * s_current_estimated[1] + torch.log(torch.e + x_features_normalized[i, 0])

#         loss1 = torch.sum((torch.abs(s_current_entropy_form - entropy_t1)))
#         loss2 = torch.sum(torch.abs(d_s_predicted - d_s_actual))
#         loss_3 = torch.sum(torch.abs(s_current_entropy_form + d_s_predicted - entropy_t2))
#         loss_4 = torch.sum(torch.abs(s_current_entropy_form - t_p_estimate))

#         t_next_actual = x_features_normalized[i+1, 0]
#         p_next_actual = x_features_normalized[i+1, 1]
#         value_loss = torch.abs((p_predicted - p_next_actual) + torch.abs(t_predicted - t_next_actual))
#         entropy_loss = torch.lgamma(torch.abs(loss1) + torch.abs(loss2) + torch.abs(loss_3) + torch.abs(loss_4))
#         inductive_bias = torch.lgamma(value_loss - entropy_loss)

#         loss = entropy_loss + inductive_bias + value_loss

#         entropy_losses.append(entropy_loss.item())
#         temp_pressure_losses.append(value_loss.item())
#         total_losses.append(loss.item())
#         inductive_losses.append(inductive_bias.item())

#         if (i) % 100 == 0:
#             print("------------------------------------------------")
#             print(" ")
#             print("step ", i)
#             print("Previous Actual Everything (normalized) ", x_features_normalized[i, ])
#             print("Next Actual: T, P (normalized) ", t_next_actual, p_next_actual)
#             print("Estimated T, P (normalized) ", t_predicted, p_predicted)
#             print("Actual Entropy Change", d_s_actual)
#             print("Predicted Entropy Change", d_s_predicted)
#             print("value loss, entropy loss", value_loss, entropy_loss)

#             x_features_unnormalized_nextStep_actual = ((((x_features_normalized[i+1] - torch.e) / 10) - 0.0000000000001 * torch.sqrt(batch_norm.running_var.detach() + batch_norm.eps)) + batch_norm.running_mean.detach())
#             x_features_normalized_nextStep_predicted = x_features_normalized[i]  # just for the skeptics

#             x_features_normalized_nextStep_predicted[0], x_features_normalized_nextStep_predicted[1] = t_predicted.detach(), p_predicted.detach()
#             T_P_features_unnormalized_nextStep_predicted = ((((x_features_normalized_nextStep_predicted - torch.e) / 10) - 0.0000000000001 * torch.sqrt(batch_norm.running_var + batch_norm.eps)) + batch_norm.running_mean)[0:2]

#             print('real next step temp and pressure', x_features_unnormalized_nextStep_actual)
#             print('estimated next step temp and pressure', T_P_features_unnormalized_nextStep_predicted)
#             print(" ")
#             print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

#             # Plot the losses
#             fig, ax = plt.subplots(figsize=(10, 5))
#             ax.plot(total_losses, label='Total Loss')
#             ax.plot(entropy_losses, label='Entropy of Prediction vs Actual (EoPvA)')
#             ax.plot(temp_pressure_losses, label='Predictive T,P Loss')
#             ax.plot(inductive_losses, label="Inductive Bias")

#             ax.set_xlabel('Iterations')
#             ax.set_ylabel('Log')
#             ax.set_yscale('log')
#             ax.legend()

#             # Save the plot to a file
#             plt.savefig('./inference_loss_plot.png')
#             import matplotlib.pyplot as plt
# import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store temperature, pressure, and index of maximum out values
temperatures = []
pressures = []
max_out_indices = []
import numpy as np

with torch.no_grad():
    out_values = [[] for _ in range(20)]  # Create a list to store the out values for each driver
    temperatures = []
    pressures = []
    for i in range(x_features_normalized.shape[0]):
        if i == len(x_features_normalized) - 1:
            continue  # Skip the last step
        if i <= 25:
            continue  # Skip the first 25 steps
        x_prev_24 = x_features_normalized[i-24:i, :]
        s_current_estimated = xfmr.u1_forward(x_prev_24[-2:, 0:2])
        x_prev_24_pe = xfmr.pe_forward(x_prev_24)
        out = xfmr(x_prev_24_pe, s_current_estimated)
        t_next_actual = x_features_normalized[i+1, 0]
        p_next_actual = x_features_normalized[i+1, 1]
        temperatures.append(t_next_actual.item())
        pressures.append(p_next_actual.item())
        # Apply exponential function to the out values
        exp_out_values = [torch.exp(1 + out[j]) for j in range(20)]
        # Store the exponential out values for each driver
        for j in range(20):
            out_values[j].append(exp_out_values[j].item())
        print(i)
        if i % 20 == 0:
            # Create separate plots for each driver
            for r in range(20):
                fig, ax = plt.subplots(figsize=(10, 8))
                # Apply logarithmic scale to the out values
                log_out_values = np.exp(out_values[r]) + 1
                # Get the minimum and maximum log-scaled out values for the current driver
                vmin = min(log_out_values)
                vmax = max(log_out_values)
                scatter = ax.scatter(temperatures, pressures, c=log_out_values, cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_xlabel('Temperature')
                ax.set_ylabel('Pressure')
                ax.set_title(f'exp-scaled Exponential Out Value of Driver {r+1} Over Time')
                # Add a colorbar to the plot
                cbar = fig.colorbar(scatter)
                cbar.set_label('exp-scaled Exponential Out Value')
                # Save the plot to a file
                plt.savefig(f'{r+1}_exp.png')
                plt.close()