from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from model import MyModel
from utils import load_data, smooth_decrease_weights

model_name = 'bert-base-uncased'
task_name = 'mrpc'  # Example GLUE task
max_length = 128
batch_size = 16  # Smaller batch size
learning_rate = 1e-4
num_epochs = 5  # Only one epoch for testing
decrease_factor = 0.95  # Adjust this factor for the weight decrease

# Load a smaller subset of data for quick testing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_loader = load_data(task_name, model_name, max_length, batch_size)

# Initialize the model
model = MyModel()  # Replace with your model's class name if different
model.train()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Gradually decrease weights of low-rank matrices
        smooth_decrease_weights(model, decrease_factor)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the model if needed
# torch.save(model.state_dict(), 'model.pth')
