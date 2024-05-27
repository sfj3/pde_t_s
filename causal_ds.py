import torch
from torch import Tensor
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
# wee if you can learn a const to rescale entropy, params may also need a global search
#also need to show the map by weights
torch.autograd.set_detect_anomaly(True)
# Data stuff
df = pd.read_pickle('sorted.pkl')  # Ensure this is already sorted by time
df['Temperature'] = df['Temperature'] +273
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
feature_indices = [0,1,2,3,4,5,6,7,8,9]  # Example indices
batch_norm = nn.BatchNorm1d(len(feature_indices), affine=False) 
x_features = x[:, feature_indices]
x_features_normalized = torch.e+ batch_norm((x_features.type('torch.FloatTensor') + 0.0000000000001) * 10.00000000000)
x_features_unnormalized = ((((x_features_normalized-torch.e) / 10)-0.0000000000001 * torch.sqrt(batch_norm.running_var + batch_norm.eps)) + batch_norm.running_mean)
# print("Original tensor:", x) #for sanity
# print("norm" ,x_features_normalized)
# print("Unnormalized tensor:", x)




#positional encoding for d prime
class PositionalEncoding_2(nn.Module):
    def __init__(self):
        num_layers = 10
        super().__init__()
        self.num_layers = num_layers
        for _ in range(num_layers):
            self.scale_ = nn.Parameter(torch.ones(10))
            self.bias_ = nn.Parameter(torch.zeros(10))

    def forward(self, x: Tensor) -> Tensor:
        num_layers = self.num_layers
        for _ in range(num_layers):
            x = self.scale_*x + self.bias_
            x = torch.sigmoid(x)
        return x
        

    def inverse(self, output: Tensor) -> Tensor:
        num_layers = self.num_layers
        x = output
        for _ in reversed(range(num_layers)):
            x = torch.log(x/ (1 - x))
            x = (x - self.bias_)/self.scale_
        return x


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
         
        

   
         



# f_1 = PositionalEncoding_1() # 10 layers       we may or may not actually need these..       
# f_2 = PositionalEncoding_2() # 10 layers
xfmr = TransformerModel()
lrd = 0.000001
optimizer = optim.Adam(xfmr.parameters(), lr=lrd)

# lrd_pe = 0.00011111
# optimizer_pe = optim.Adam(f_2.parameters(), lr=lrd_pe)

entropy_losses = []
temp_pressure_losses = []
total_losses = []
inductive_losses = []
fig, ax = plt.subplots(figsize=(10, 5))
#put together a few...
x_features_normalized = torch.cat((x_features_normalized,x_features_normalized,x_features_normalized,x_features_normalized,x_features_normalized,x_features_normalized,x_features_normalized),dim=0)

num_epochs = 10  # Specify the number of epochs you want to run
prev_value_loss = 0
prev_entropy_loss = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    for i in range(x_features_normalized.shape[0]):
        if i == len(x_features_normalized)-1: continue # because we are estimating the next step
        if i<=25: continue # because we are training on the previous 24 hours of fft(EVERYTHING) in the data using a transformer.

        entropy_t2 = torch.log(torch.e+x_features_normalized[i+1,0])- 8.314 *torch.log(torch.e+x_features_normalized[i+1,1]/2) # divide by two so you dont get negative numbers, theyve already been batch normed
        entropy_t1 = torch.log(torch.e+x_features_normalized[i,0])- 8.314*torch.log(torch.e+x_features_normalized[i,1]/2)
        d_s_actual = entropy_t2-entropy_t1

        # u_t = f_1(x_features_normalized[i,0:2])
        # g_t= f_2(x_features_normalized[i,:])
        x_prev_24 = x_features_normalized[i-24:i,]



        s_current_estimated = xfmr.u1_forward(x_prev_24[-2:,0:2]) #which you can take the sum of and bias towards the entropy, as well as make sure to feed this through the other nn
        
        x_prev_24_pe = xfmr.pe_forward(x_prev_24)
        out = xfmr(x_prev_24_pe,s_current_estimated) # make sure this gets added to the transformer architecture. 

        
        

        x_current = x_features_normalized[i]
        t_p_estimate = torch.abs(torch.sum(out.view(2,10) * x_current,axis=-1)) # this is the temperature and pressure estimate
        d_s_predicted = t_p_estimate[-1]
        s_new_estimate = d_s_predicted #+ s_current_estimated[0]-8.314*s_current_estimated[1]
        s_current_inverse = xfmr.u1_inverse(s_new_estimate)
        t_predicted = s_new_estimate + 8.314*torch.log(torch.e+x_features_normalized[i,1]/2)
        p_predicted = s_new_estimate - torch.log(torch.e+x_features_normalized[i,0])
        # loss = torch.abs()
        # entropy loss should be entropy at a point estimated with the u1_forward
        #the difference will give you the entropy thats how you put them together, ie
        #s_current_estimated - s_current actual, bijective knowing how its put together
        s_current_entropy_form = s_current_estimated[0]-8.314*s_current_estimated[1]+torch.log(torch.e+x_features_normalized[i,0]) #this is the estimated current entropy, now the error for this is
        
        #these four losses are the entropy losses
        loss1 = torch.sum((torch.abs(s_current_entropy_form - entropy_t1)))
        
        #delta s current estimated - delta s current actual
        loss2 = torch.sum(torch.abs(d_s_predicted - d_s_actual)) 
        #s next - s next actual, where s next is s_current_estimated + delta_s_current_estimated
        loss_3 = torch.sum(torch.abs(s_current_entropy_form + d_s_predicted - entropy_t2)) 
        # the inverse of s_current entropy form should equate to the temperature and pressure
        loss_4 = torch.sum(torch.abs(s_current_entropy_form - t_p_estimate))
        #s_current_inverse 
        #those are your four equations for the loss function
        #t and p next are the standard loss
        t_next_actual = x_features_normalized[i+1,0]
        p_next_actual = x_features_normalized[i+1,1]
        everything_actual_prev = x_features_normalized[i,]
        value_loss = torch.lgamma((1+torch.abs((p_predicted-p_next_actual)) * (1+torch.abs(t_predicted - t_next_actual))))
        entropy_loss = torch.lgamma(torch.abs(loss1)+torch.abs(loss2)+torch.abs(loss_3)+torch.abs(loss_4))
        inductive_bias = torch.abs(value_loss.detach() - entropy_loss.detach()) #+ torch.log(torch.max(entropy_loss.detach()/prev_entropy_loss,prev_entropy_loss/entropy_loss.detach())))
        # # inductive_bias = inductive_bias1/value_loss
        # if(inductive_bias<10**3):
        #      lrd *= 0.0001
        #      for param_group in optimizer.param_groups:
        #          param_group['lr'] = lrd

        # if (i+1) % 29 == 0:
        #             lrd *= 0.1
        #             if lrd< 0.00001: 
        #                 lrd = 0.06
        #             for param_group in optimizer.param_groups:
        #                 param_group['lr'] = lrd


        loss = 1.5*entropy_loss + 0.5*inductive_bias + value_loss #we want slight more bias for the entropy loss
        prev_value_loss = value_loss
        prev_entropy_loss = prev_entropy_loss
        # if(torch.isnan(loss)):break
        # torch.nn.utils.clip_grad_norm_(xfmr.parameters(), max_norm= 11.0) 
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.999, patience=5, verbose=True)
        # scheduler.step(loss)
        # loss1 = value_loss + entropy_loss + inductive_bias**2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        
        #now we just need to backpropagate over this loss 

        
        entropy_losses.append(entropy_loss.item())
        temp_pressure_losses.append(value_loss.item())
        total_losses.append(loss.item())
        inductive_losses.append(inductive_bias.item())
        if (i) % 100 == 0:
                print("------------------------------------------------")
                print(" ")
                print("step ",i)
                print("Previous Actual Everything (normalized) ", x_features_normalized[i,])
                print("Next Actual: T, P (normalized) ",t_next_actual,p_next_actual)
                print("Estimated T, P (normalized) ",t_predicted,p_predicted)
                print("Actual Entropy Change",d_s_actual)
                print("Predicted Entropy Change", d_s_predicted)
                print("value loss, entropy loss",value_loss,entropy_loss)

                x_features_unnormalized_nextStep_actual=((((x_features_normalized[i+1]-torch.e) / 10)-0.0000000000001 * torch.sqrt(batch_norm.running_var.detach() + batch_norm.eps)) + batch_norm.running_mean.detach())
                x_features_normalized_nextStep_predicted = x_features_normalized[i] #just for the skeptics

                x_features_normalized_nextStep_predicted[0],x_features_normalized_nextStep_predicted[1] = t_predicted.detach(),p_predicted.detach()
                T_P_features_unnormalized_nextStep_predicted=((((x_features_normalized_nextStep_predicted-torch.e) / 10)-0.0000000000001 * torch.sqrt(batch_norm.running_var + batch_norm.eps)) + batch_norm.running_mean) [0:2]
                
                
                print('real next step temp and pressure',x_features_unnormalized_nextStep_actual)
                print('estimated next step temp and pressure', T_P_features_unnormalized_nextStep_predicted)
                print(" ")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                # Clear the plot
                ax.clear()

                # Plot the losses
                ax.plot(total_losses, label='Total Loss')
                ax.plot(entropy_losses, label='Entropy of Prediction vs Actual (EoPvA)')
                ax.plot(temp_pressure_losses, label='Predictive T,P Loss')
                ax.plot(inductive_losses,label = "Inductive Bias")

                ax.set_xlabel('Iterations')
                ax.set_ylabel('Loss')
                # ax.set_yscale('log')
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
                plt.savefig(f'plots/loss_plot.png')
                torch.save(xfmr.state_dict(), f'causal_weights.pth')
        





