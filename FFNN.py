import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for progress bars

# Load and preprocess corpus
def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower().split('\n')
    return text

# Tokenize and create n-grams
def create_ngrams(sentences, n):
    ngrams = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) < n:
            continue
        for i in range(len(words) - n):
            ngrams.append((tuple(words[i:i+n-1]), words[i+n-1]))
    return ngrams

# Build vocabulary
def build_vocab(ngrams):
    word_counts = Counter()
    for context, target in ngrams:
        word_counts.update(context)
        word_counts[target] += 1
    vocab = {word: i for i, (word, _) in enumerate(word_counts.most_common())}
    return vocab

# Convert n-grams to tensor format
def encode_data(ngrams, vocab):
    X, y = [], []
    for context, target in ngrams:
        X.append([vocab[word] for word in context])
        y.append(vocab[target])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Define FFNN Model
class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size):
        super(FFNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Train model
def train_model(model, X_train, y_train, epochs=5, lr=0.01):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(epochs), desc="Training Epochs"):  # Add tqdm for epochs
        optimizer.zero_grad()
        output = model(X_train)
        loss = loss_fn(output, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Calculate perplexity
def calculate_perplexity(model, X, y):
    with torch.no_grad():
        output = model(X)
        log_probs = nn.functional.log_softmax(output, dim=1)
        loss = -log_probs[range(len(y)), y].mean()
        return torch.exp(loss).item()

# Write scores to file
def write_scores_to_file(sentences, scores, file_name):
    avg_perplexity = sum(scores) / len(scores)  # Calculate average perplexity
    with open(file_name, "w") as file:
        file.write(f"Average Perplexity: {avg_perplexity}\n")  # Write average perplexity first
        for sentence, score in zip(sentences, scores):
            file.write(f"{sentence}\t{score}\n")

# Main execution
if __name__ == "__main__":
    n = int(input("Enter n for n-gram: "))
    train_texts = load_corpus("pride_and_prejudice.txt")
    train_sentences, test_sentences = train_test_split(train_texts, test_size=1000, random_state=42)
    
    ngrams = create_ngrams(train_sentences, n)
    vocab = build_vocab(ngrams)
    X_train, y_train = encode_data(ngrams, vocab)
    
    model = FFNNLanguageModel(len(vocab), embed_dim=50, hidden_dim=100, context_size=n-1)
    train_model(model, X_train, y_train)
    
    train_perplexity = calculate_perplexity(model, X_train, y_train)
    print(f"Training Perplexity: {train_perplexity}")
    
    # Calculate perplexity for each sentence with tqdm
    scores = [calculate_perplexity(model, X_train[i].unsqueeze(0), y_train[i].unsqueeze(0)) 
              for i in tqdm(range(len(y_train)), desc="Calculating Perplexity")]
    
    write_scores_to_file(train_sentences[:len(scores)], scores, "train_scores.txt")