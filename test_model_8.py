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

# Define the TransformerModel class (same as in the provided code)
class TransformerModel(nn.Module):
    def __init__(self, n_features, n_hiddens, n_layers, n_heads, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding_greek = nn.Linear(5, n_hiddens)
        encoder_layers_greek = nn.TransformerEncoderLayer(n_hiddens, n_heads, n_hiddens, dropout)
        self.transformer_encoder_greek = nn.TransformerEncoder(encoder_layers_greek, n_layers)
        self.decoder_greek = nn.Linear(n_hiddens, 6)

    def forward(self, src):
        src = torch.abs(torch.fft.fft(src,dim=0))
        grk = src
        src_grk = self.embedding_greek(grk)
        output_grk = self.transformer_encoder_greek(src_grk)
        output_grk = self.decoder_greek(output_grk[-1])
        return output_grk

# Load and process the DataFrame (same as in the provided code)
df = pd.read_pickle('sorted.pkl')  # Ensure this is already sorted by time
df['Temperature']  = (df['Temperature'] + 273)
df['Relative Humidity']  = df['Relative Humidity']
df['Pressure']  = df['Pressure']
x = torch.tensor(df[['Temperature', 'Pressure', 'Relative Humidity', 'DNI', 'Wind Speed']].values, dtype=torch.float32)

# Parameters (same as in the provided code)
n_features = 1
n_hiddens = 1
n_layers = 1
n_heads =  1
n_epochs = 60
batch_size = 48
learning_rate = 0.0003460001

# Create the model
model = TransformerModel(n_features, n_hiddens, n_layers, n_heads)

# Load the saved model state
model.load_state_dict(torch.load('latest_epoch_1.pth'))

# Set the model to evaluation mode
model.eval()

# Debug mode testing
with torch.no_grad():
    for i in range(0, len(x), batch_size):
        batch = x[i:i+batch_size]
        outputs_grk = model(batch[:-1])
        actual = batch[-1,]

        temp_1,temp_2,p_1,p_2,rh_2,rh_1,i_2,i_1,v_1,v_2 = batch[-2,0],batch[-1,0],batch[-2,1],batch[-1,1],batch[-2,2],batch[-1,2],batch[-2,3],batch[-1,3],batch[-2,4],batch[-1,4]

        alpha,beta,gamma,xi,mu,theta = outputs_grk

        # Perform the same calculations as in the training loop
        p_pred = p_1
        t_pred = temp_1
        r_pred = (v_1*gamma)**-1 * (mu - alpha * t_pred **-1 - beta * i_1 - xi*theta) + rh_1 
        v_pred = (r_pred*gamma)**-1 * (mu - alpha * t_pred **-1 - beta * i_1 - xi*theta) + v_1 
        i_pred = beta ** -1 * (mu - alpha/t_pred - gamma * v_pred * r_pred - xi*theta) + i_1 
        t_pred = (mu - beta * i_pred - gamma * v_1 * r_pred - xi *(theta))**-1 + temp_1
        p_pred =torch.exp((mu - torch.log(t_pred/temp_1))/8.314 + torch.log(p_1))
        r_pred = (v_1*gamma)**-1 * (mu - alpha * t_pred **-1 - beta * i_1 - xi*theta) + rh_1
        v_pred = (r_pred*gamma)**-1 * (mu - alpha * t_pred **-1 - beta * i_1 - xi*theta) + v_1 
        i_pred = beta ** -1 * (mu - alpha/t_pred - gamma * v_pred * r_pred - xi*theta) + i_1 
        t_pred = (mu - beta * i_pred - gamma * v_1 * r_pred - xi *(theta))**-1 + temp_1
        p_pred =torch.exp((mu - torch.log(t_pred/temp_1))/8.314 + torch.log(p_1)) 
        x_prime = torch.cat((t_pred.unsqueeze(0),p_pred.unsqueeze(0),r_pred.unsqueeze(0),i_pred.unsqueeze(0),v_pred.unsqueeze(0)))

        print("Predicted values:", x_prime)
        print("Actual values:", batch[-1, :])
        print("---")

# You can now modify and experiment with the code as needed in this debug environment