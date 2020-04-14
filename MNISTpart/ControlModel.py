from torch import nn
import torch.nn.functional as F
import torch
from torch import optim
from MNISTpart import MNISTdata

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        out = F.dropout(x, p=0.2)
        out = self.fc1(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.2)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

class ControlModel():
    def __init__(self):
        self.loss_function = nn.CrossEntropyLoss()
        self.model = self.trainModelOnMNIST()
        self.optimiser = optim.SGD(self.model.parameters(),lr=0.1, momentum=0.5)

    def trainModelOnMNIST(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = Model(784, 1200, 10).to(device)
        loss_function = self.loss_function
        optimiser = optim.SGD(model.parameters(),lr=0.1, momentum=0.5)
        data = MNISTdata.MNISTdata()
        trainloader = data.trainloader

        for epoch in range(2):
          for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimiser.step()
        print('Control Model trainning completed')

        return model
