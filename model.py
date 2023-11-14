import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(in_features=10, out_features=50)  # Adjusted to match input features
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=50, out_features=10)   # Another example layer


    def forward(self, x):
        # Define the forward pass
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
