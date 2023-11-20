import torch
import torch.nn as nn
import math

class ReLoRaLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(ReLoRaLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Original weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.Tensor(rank, in_features))
        self.lora_B = nn.Parameter(torch.Tensor(out_features, rank))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input):
        # Low-rank approximation
        lora_output = torch.mm(self.lora_B, torch.mm(self.lora_A, input.T)).T
        # Combine with original linear transformation
        return nn.functional.linear(input, self.weight, self.bias) + lora_output

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
