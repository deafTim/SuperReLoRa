import argparse
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from model import MyModel
from utils import load_data, smooth_decrease_weights

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SuperReLoRa.")

    # Add arguments
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model identifier")
    parser.add_argument("--task_name", type=str, default="mrpc", help="GLUE task name")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--decrease_factor", type=float, default=0.95, help="Factor for weight decrease")

    # Parse arguments
    args = parser.parse_args()
    return args


    # Use args to access the command-line arguments
    # Example: print(args.model_name)

if __name__ == "__main__":
    args = parse_args()
    # Load data
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    train_loader = load_data(args.task_name, args.model_name, args.max_length, args.batch_size)

    # Initialize the model
    model = MyModel()  # Replace with your model's class name if different
    model.train()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
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
            smooth_decrease_weights(model, args.decrease_factor)

        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item()}")

    # Save the model if needed
    # torch.save(model.state_dict(), 'model.pth')
