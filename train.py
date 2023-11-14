import torch
import torch.nn as nn
import torch.optim as optim
from model import MyModel
from utils import smooth_restart_weights

# Hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 100

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = MyModel().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loader (example with random data)
# Replace this with your actual data loader
train_loader = [(torch.randn(batch_size, 10), torch.randint(0, 2, (batch_size,))) for _ in range(100)]

# Training loop
for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Example: Smooth restart of weights at the end of each epoch
    if epoch % 2 == 0:  # Example condition, adjust as needed
        smooth_restart_weights(model, alpha=0.1)

print("Training finished.")
