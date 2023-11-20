import torch
import torch.nn as nn
from transformers import BertModel
import math

class MyModel(nn.Module):
    def __init__(self, lora_dim=32):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Assuming the hidden size of BERT base model is 768
        hidden_size = 768

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.Tensor(lora_dim, hidden_size))
        self.lora_B = nn.Parameter(torch.Tensor(hidden_size, lora_dim))

        # Output layer
        self.output = nn.Linear(hidden_size, 2)  # Example for binary classification

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        nn.init.zeros_(self.output.bias)

    def forward(self, input_ids, attention_mask):
        # Get the output from the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        # Apply low-rank approximation
        lora_output = torch.matmul(sequence_output, self.lora_A.T)  # Transpose lora_A
        lora_output = torch.matmul(lora_output, self.lora_B.T)

        # Combine BERT output and low-rank approximation
        combined_output = sequence_output + lora_output

        # Pass through the output layer
        logits = self.output(combined_output[:, 0, :])

        return logits
