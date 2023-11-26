import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt

import numpy as np
import time
import scipy.io
import pandas as pd

df = pd.read_csv("upsample.csv")
x1 = df['GLIR'].to_numpy()
x1 = x1.reshape(len(x1),1)
y1 = df['Qo'].to_numpy()
x = torch.from_numpy(x1)
x = torch.tensor(x,dtype=torch.float)

y = torch.from_numpy(y1)
y = torch.tensor(y,dtype=torch.float)

# this is one way to define a network
class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden1, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden, bias=True)    # hidden layer0
        self.hidden1 = nn.Linear(n_hidden, n_hidden1, bias=True)   # hidden layer1
        self.predict = nn.Linear(n_hidden1, n_output, bias=True)   # output layer

    def forward(self, x):
        x = F.softmax(self.hidden(x))      # softmax activation function for hidden layer0
        x = F.relu(self.hidden1(x))        # relu activation function for hidden layer1
        x = self.predict(x)                # linear output
        return x

net = Net(n_feature=1, n_hidden=50, n_hidden1=50, n_output=1)     # define the network
print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
loss_func = nn.MSELoss()  # this is for regression mean squared loss


"""def plot(t,x,y,predict,losses,dt):
    clear_output(True)
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.title('iteration %s'% (t))
    plt.plot(times.data.numpy(),y.data.numpy()[0:1000], color = "orange")
    
    plt.text(0, -2.5, 'scan time = %.4f' % dt, fontdict={'size': 12, 'color':  'red'})
    plt.text(0, -2.1, 'Loss = %.4f' % loss.data.numpy(),
            fontdict={'size': 12, 'color':  'red'})
    
    plt.plot(range(0,len(predict)),predict, color = "green")
    #plt.plot(times.data.numpy(),predict.data.numpy()[0:1000], color = "green")
    
    plt.subplot(122)
    plt.title('Loss')
    plt.plot(losses)
    
    plt.show()"""

predictiond=[]
losses=[]
N = 35      # window size

times = torch.unsqueeze(torch.linspace(0, 1000,1000), dim=1)
for t in range(1000):
    start = time.time()  
    
    prediction = net(x[0:1000])     # input x and predict based on x

    loss = loss_func(prediction[t:t+N], y[t:t+N])     # must be (1. nn output, 2. target)
    losses.append(loss.data.numpy())
    
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    duration = time.time()-start
    predictiond.append(prediction.data.numpy()[0][-1])

    pred = prediction.data.numpy()[0]

    #print('value',np.shape(predictiond))
    #print('value',predictiond)
    
    # plot and show learning process
    #if t % 20==0:
        #plt.plot(t,x,y,predictiond,losses,duration)
        #print(net.predict.weight)
        #print(np.mean(losses),np.std(losses))
#plt.plot(t,x,y,predictiond,losses,duration)

print('value',np.shape(predictiond))
print('value',predictiond)
timee = []
t = range(0,len(predictiond))
for T in t:
    timee.append(T)

print('time',np.shape(T))
print('time',timee)

plt.plot(timee,predictiond)

"""print('X VALUE:',x[1000:])
y_test = net.forward(x[1000:])
print('Y VALUE:',y_test)
#plt.plot(times.data.numpy(), y_test.data.numpy(), times.data.numpy(), y.data.numpy()[1000:2000], lw=2)
loss = loss_func(y_test, y[1000:])
print(loss)"""
