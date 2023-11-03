import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

#Generating random data of 100 samples where each sample is of 2 values  Eg: each sample [0.234, 0.567]
X_train = torch.rand(100,2, dtype=torch.float)
y_train = torch.randint(2, size=(100,1), dtype = torch.float)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, layer1_size, layer2_size, outlayer):
        super(NeuralNetwork, self).__init__()

        self.d1 = nn.Linear(input_size, layer1_size)
        self.d2 = nn.Linear(layer1_size, layer2_size)
        self.output = nn.Linear(layer2_size, outlayer)
    
    def forward(self,x):
        d1_out = nn.functional.relu(self.d1(x))
        d2_out = nn.functional.relu(self.d2(d1_out))
        d3_out = nn.functional.sigmoid(self.output(d2_out))

        return d3_out

model = NeuralNetwork(input_size = 2, layer1_size = 5, layer2_size = 3, outlayer = 1)

loss_criteria = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0025)

epochs = 10
batch_size = 4

for epoch in range(epochs):

    #just a progress bar for the training process
    progress_bar = tqdm(range(0, len(X_train), batch_size), position=0, leave=True)

    for i in progress_bar:
        inputs = X_train[i : i + batch_size]
        labels = y_train[i : i + batch_size]

        optimizer.zero_grad() #Zero out the gradients

        model_output = model(inputs)
        loss = loss_criteria(model_output, labels)

        loss.backward() #Backpropagation
        optimizer.step() #For updating model parameters

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

