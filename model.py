import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Common Dataset Class
class NWPDataset(Dataset):
    def __init__(self, text, n):
        self.n = n
        self.ngrams = self._create_ngrams(text)
        self.vocab = self._create_vocab(text)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def _create_ngrams(self, text):
        words = text.split()
        ngrams = []
        for i in range(len(words) - self.n):
            ngrams.append((words[i:i+self.n], words[i+self.n]))
        return ngrams

    def _create_vocab(self, text):
        words = text.split()
        vocab = list(set(words))
        vocab.append("<PAD>")  # Add padding token
        return vocab

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        context, target = self.ngrams[idx]
        context_idx = [self.word_to_idx[word] for word in context]
        target_idx = self.word_to_idx[target]
        return torch.tensor(context_idx, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)


# Base Language Model Class
class BaseLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dataset):
        super(BaseLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.dataset = dataset  # Store dataset reference
        self.pad_idx = dataset.word_to_idx["<PAD>"]

    def forward(self, x):
        raise NotImplementedError

    def predict(self, context, k=1):
        self.eval()
        with torch.no_grad():
            context_idx = [self.dataset.word_to_idx.get(word, self.pad_idx) for word in context]
            while len(context_idx) < self.dataset.n:
                context_idx.insert(0, self.pad_idx)  # Left-pad with <PAD>

            context_idx = torch.tensor(context_idx, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            logits = self.forward(context_idx)
            probs = torch.softmax(logits, dim=-1)
            top_k = torch.topk(probs, k=k, dim=-1)
            top_k_words = [self.dataset.idx_to_word[idx.item()] for idx in top_k.indices[0]]
        return top_k_words


# Feed Forward Neural Network Language Model
class FFNNLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n, dataset):
        super(FFNNLanguageModel, self).__init__(vocab_size, embedding_dim, hidden_dim, dataset)
        self.n = n
        self.embedding_dim = embedding_dim  # Store embedding dimension
        self.fc1 = nn.Linear(n * embedding_dim, hidden_dim)  # Input must match reshaped embedding
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)  # Expected shape: (batch_size, n, embedding_dim)
        batch_size, seq_length, emb_dim = embedded.shape
        embedded = embedded.view(batch_size, seq_length * emb_dim)  # Flatten properly
        hidden = torch.relu(self.fc1(embedded))
        logits = self.fc2(hidden)
        return logits


# Training Function
def train_model(model, dataset, epochs=10, lr=0.001, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            logits = model(context)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")


# Example Usage
text = "This is a simple example of a text for next word prediction."
n = 3  # n-gram size

dataset = NWPDataset(text, n)
ffnn_model = FFNNLanguageModel(len(dataset.vocab), embedding_dim=10, hidden_dim=20, n=n, dataset=dataset)
train_model(ffnn_model, dataset, epochs=10)

# Predict next word
context = ["simple","example"]
print("FFNN Prediction:", ffnn_model.predict(context, k=1))