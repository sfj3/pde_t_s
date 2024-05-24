# potentially to do for model 7: try global search model? try different criterions, gradient smoothing, take log of loss or loss component, loss component balancing... play with LR scheduler
#also see if you really should be estimating the second time step for anything but temperature! test the model in different dynamical scenarios it hasnt seen, build a causal model to validate this
#de is modified from https://www.sciencedirect.com/science/article/abs/pii/S0020722507000900?via%3Dihub, math condition that can be satisfied: measure for how likely it is to operate beyond the training data, age and climate change example..., there should be a condition that can be satisfied, what is the probability that its wrong
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
# Convert 'Temperature' and 'Pressure' to PyTorch tensors
x = torch.tensor(df[['Temperature', 'Pressure', 'Relative Humidity', 'DNI', 'Wind Speed']].values, dtype=torch.float32)

class TransformerModel(nn.Module):
    def __init__(self, n_features, n_hiddens, n_layers, n_heads, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(5, n_hiddens)
        encoder_layers = nn.TransformerEncoderLayer(n_hiddens, n_heads, n_hiddens, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)#try lstm instead of transformer here
        self.decoder = nn.Linear(n_hiddens, 2)

        self.embedding_greek = nn.Linear(5, n_hiddens)
        encoder_layers_greek = nn.TransformerEncoderLayer(n_hiddens, n_heads, n_hiddens, dropout)
        self.transformer_encoder_greek = nn.TransformerEncoder(encoder_layers_greek, n_layers)
        self.decoder_greek = nn.Linear(n_hiddens, 6)

    def forward(self, src):
        grk = src

        src = self.embedding(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1]) # just pressure
        
        src_grk = self.embedding_greek(grk)
        output_grk = self.transformer_encoder_greek(src_grk)
        output_grk = self.decoder_greek(output_grk[-1])



        return output,output_grk

# Parameters
n_features = 48
n_hiddens = 12
n_layers = 4
n_heads =  12
n_epochs = 300
batch_size = 48
learning_rate = 0.001

# Create the model
model = TransformerModel(n_features, n_hiddens, n_layers, n_heads)

# Loss function and optimizer
criterion =  nn.CosineSimilarity(dim=0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
if not os.path.exists('plots'):
    os.makedirs('plots')

loss_values = []
loss_lcec_values = []
diff_loss_values = []
fig, ax = plt.subplots(figsize=(10, 5))

for epoch in range(n_epochs):
    print(epoch)
    for i in range(0, len(x), batch_size):

        batch = x[i:i+batch_size]

        # Forward pass
        t_p,outputs_grk = model(batch[:-1])
        t = t_p[0]
        p = t_p[1]
        actual = batch[-1,]
        


        #now we need to create the loss of the greeks to the entropy

        temp_1,temp_2,p_1,p_2,rh_2,rh_1,i_2,i_1,v_2,v_1 = batch[-2,0],batch[-1,0],batch[-2,1],batch[-1,1],batch[-2,2],batch[-1,2],batch[-2,3],batch[-1,3],batch[-2,4],batch[-1,4]
        
        #now the entropy loss with the learned constant entropy
        #learned constant entropy
        alpha,beta,gamma,xi,mu,theta = outputs_grk
        t_pred = t
        p_pred = p
        t_pred_1 = t * alpha * (mu - beta * i_1 - gamma * v_1 * rh_1 - xi *(theta))**-1 #before this was rh2-rh1 instead of theta, you could use rh1 as well to try, also maybe you can predict all the other variables this way off of each other!
        r_pred = (v_1*gamma)**-1 * (mu - alpha * temp_1 **-1 - beta * i_1 - xi*theta)
        v_pred = (rh_1*gamma)**-1 * (mu - alpha * temp_1 **-1 - beta * i_1 - xi*theta)
        i_pred = beta ** -1 * (mu + alpha/temp_1 - gamma * v_1 * rh_1 - xi*theta)
        t_pred = t * alpha * (mu - beta * i_pred - gamma * v_pred * r_pred - xi *(theta))**-1
        x_prime = torch.cat((t_pred.unsqueeze(0),p_pred.unsqueeze(0),r_pred.unsqueeze(0),i_pred.unsqueeze(0),v_pred.unsqueeze(0)))
        #all_pred  = torch.cat((batch[:-1,],x_prime.unsqueeze(0)),0) # maybe comment out
        # and the loss on the temperature: 
        d_entropy_pt = torch.log(temp_2/temp_1)  - 8.314 * torch.log(p_2/p_1) # the actual delta entropy
        d_entropy_lce = (alpha/temp_2 + beta * i_2 + gamma * v_2 * rh_2 + xi * (rh_2)) - (alpha/temp_1 + beta * i_1 + gamma * v_1 * rh_1 + xi * (rh_1)) 
        #loss for the learned constants to be accurate (learned constant entropy constants loss)
        loss_lcec = torch.abs(rh_2-r_pred)+torch.abs(p_2-p_pred)+torch.abs(t_pred_1-t_pred)+(torch.abs(d_entropy_lce-d_entropy_pt))+(torch.abs(mu-d_entropy_pt))+(torch.abs(theta - (rh_2 - rh_1))) #before there was no constant theta
        #now we can compute the temperature and add that to the final loss
        # Backward pass and optimization
        optimizer.zero_grad()

        loss =   (loss_lcec) + criterion(x_prime,batch[-1,]) # see if focusing more on temperature loss and less on sporadic losses like wind prediction helps, ie use the transformer to estimate the current state for nontemp variables
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
        optimizer.step()
        print("outputs ",x_prime)
        print("actual ", batch[-1, :])
        # print("loss",loss,"lcec loss",loss_lcec,"l1 loss " ,loss - loss_lcec)
        # if (i+1) % 1800 == 0:
        #     learning_rate *= 0.99999
        #     if learning_rate < 0.00001:
        #         learning_rate = 0.00001
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = learning_rate

        # Store the loss values
        loss_values.append(loss.item())
        loss_lcec_values.append(loss_lcec.item())
        diff_loss_values.append((loss - loss_lcec).item())
        if (i) % 2000 == 0:
            # Clear the plot
            ax.clear()

            # Plot the losses
            ax.plot(loss_values, label='Loss')
            ax.plot(loss_lcec_values, label='LCEC Loss')
            ax.plot(diff_loss_values, label='l1 Loss')

            ax.set_xlabel('Iterations')
            ax.set_ylabel('Loss')
            ax.set_yscale('log')
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
