import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import argparse
import os
from sklearn.model_selection import train_test_split

# Create n-grams
def create_ngrams(sentences, n):
    ngrams = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) < n:
            continue
        for i in range(len(words) - n):
            ngrams.append((tuple(words[i:i+n-1]), words[i+n-1]))
    return ngrams

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

# Build vocabulary
def build_vocab(ngrams):
    word_counts = Counter()
    for context, target in ngrams:
        word_counts.update(context)
        word_counts[target] += 1
    vocab = {word: i for i, (word, _) in enumerate(word_counts.most_common())}
    idx_to_word = {i: word for word, i in vocab.items()}  # Reverse mapping
    return vocab, idx_to_word

# Tokenize input sentence
def tokenize_sentence(sentence, vocab, context_size):
    words = sentence.lower().split()
    if len(words) < context_size:
        raise ValueError(f"Input sentence must have at least {context_size} words.")
    context = words[-context_size:]  # Use the last `context_size` words as context
    return [vocab.get(word, vocab["<unk>"]) for word in context]  # Handle unknown words

# Generate top-k next word candidates
def generate_next_words(model, context, idx_to_word, k):
    with torch.no_grad():
        context_tensor = torch.tensor([context], dtype=torch.long)
        output = model(context_tensor)
        probs = torch.softmax(output, dim=1).squeeze()
        top_k_probs, top_k_indices = torch.topk(probs, k)
        return [(idx_to_word[idx.item()], prob.item()) for idx, prob in zip(top_k_indices, top_k_probs)]

# Train the model
def train_model(corpus_path, n, embed_dim=50, hidden_dim=100, epochs=5, lr=0.01, model_save_path="ffnn_model.pt"):
    # Load corpus
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read().lower().split('\n')
    
    # Split into training and validation sets
    train_texts, _ = train_test_split(text, test_size=1000, random_state=42)
    
    # Create n-grams and build vocabulary
    ngrams = create_ngrams(train_texts, n)
    vocab, idx_to_word = build_vocab(ngrams)
    
    # Encode data
    X, y = [], []
    for context, target in ngrams:
        X.append([vocab[word] for word in context])
        y.append(vocab[target])
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    
    # Initialize model
    model = FFNNLanguageModel(len(vocab), embed_dim, hidden_dim, context_size=n-1)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    return model, vocab, idx_to_word

# Load pretrained model
def load_pretrained_model(model_path, vocab_size, embed_dim, hidden_dim, context_size):
    model = FFNNLanguageModel(vocab_size, embed_dim, hidden_dim, context_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Language Model Word Generator")
    parser.add_argument("lm_type", type=str, help="Type of language model (f for FFNN)")
    parser.add_argument("corpus_path", type=str, help="Path to the corpus file")
    parser.add_argument("k", type=int, help="Number of candidate words to generate")
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    # Validate arguments
    if args.lm_type != "f":
        raise ValueError("Only FFNN language model (f) is supported.")
    if not os.path.exists(args.corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus_path}")

    # Train the model if --train flag is set
    if args.train:
        print("Training the model...")
        train_model(args.corpus_path, n=4, model_save_path="ffnn_model.pt")
        return

    # Load corpus and build vocabulary
    with open(args.corpus_path, 'r', encoding='utf-8') as f:
        text = f.read().lower().split('\n')
    ngrams = create_ngrams(text, n=4)  # Assuming n=4 for trigram model
    vocab, idx_to_word = build_vocab(ngrams)

    # Load pretrained model
    model_path = "ffnn_model.pt"  # Path to the pretrained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    model = load_pretrained_model(model_path, len(vocab), embed_dim=50, hidden_dim=100, context_size=3)

    # Interactive prompt
    while True:
        try:
            sentence = input("Input sentence: ").strip()
            if not sentence:
                print("Please enter a valid sentence.")
                continue

            # Tokenize input sentence
            context = tokenize_sentence(sentence, vocab, context_size=3)

            # Generate top-k next words
            candidates = generate_next_words(model, context, idx_to_word, args.k)

            # Print results
            print("Output:")
            for word, prob in candidates:
                print(f"{word}\t{prob:.4f}")

        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()