import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
    sample_texts = ["This is a sample text.", "Here is another one."]
    sample_labels = [0, 1]

    encodings = tokenizer(sample_texts, truncation=True, padding=True, max_length=128)
    dataset = CustomDataset(encodings, sample_labels)

    # Example usage without print
    _ = dataset[0]  # Accessing the first item to demonstrate usage
