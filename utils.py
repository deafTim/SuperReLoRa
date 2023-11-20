import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def smooth_decrease_weights(model, decrease_factor=0.99):
    """
    Gradually decreases the weights of the low-rank matrices in the ReLoRa layers of the model.

    Parameters:
    model (torch.nn.Module): The neural network model with ReLoRa layers.
    decrease_factor (float): The factor by which the weights are decreased in each call. Default is 0.99.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                # Scale down the weights
                param.data *= decrease_factor




class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(tokenizer, max_length, batch_size):
    # Load IMDb dataset
    dataset = load_dataset("imdb")
    train_data = dataset['train']

    # Create a Dataset
    train_dataset = TextDataset(train_data, tokenizer, max_length)

    # Create a DataLoader
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return data_loader

# Example usage
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# data_loader = load_data(tokenizer, max_length=128, batch_size=32)
