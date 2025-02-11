import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import argparse
import os
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from tqdm import tqdm
from typing import List

# Download NLTK data
nltk.download('punkt')

# Tokenizer Class
class Tokenizer:
    def __init__(self):
        url_regex_pattern = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9@:%_\\+.~#?&\/=]*)?'
        hashtag_regex_pattern = r'#\w+'
        mentions_regex_pattern = r'@\w+'
        percentage_regex_pattern = r'\d+\s*\%'
        range_regex_pattern = r'\d+\s*[-–]\s*\d+'
        email_regex_pattern = r"\S+@\S+\.\S+"
        self.place_holders = [
            ["<URL>", url_regex_pattern],
            ["<HASHTAG>", hashtag_regex_pattern],
            ["<MENTION>", mentions_regex_pattern],
            ["<PERCENTAGE>", percentage_regex_pattern],
            ["<RANGE>", range_regex_pattern],
            ["<MAILID>", email_regex_pattern]
        ]
        self.multi_word_tokenizer = MWETokenizer([('<URL>',), ('<HASHTAG>',), ('<MENTION>',), ('<PERCENTAGE>',), ('<RANGE>',), ('<MAILID>',)], separator='')

    def tokenize(self, text: str) -> List[List[str]]:
        for substitution, pattern in self.place_holders:
            text = re.sub(pattern, substitution, text)
        return [self.multi_word_tokenizer.tokenize(word_tokenize(sentence)) for sentence in sent_tokenize(text)]
    
    def split_into_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

# FFNN Language Model Class
class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size):
        super(FFNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, context_size, embed_dim)
        x = x.view(x.shape[0], -1)  # Flatten to (batch_size, context_size * embed_dim)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Language Model Trainer and Evaluator
class LanguageModel:
    def __init__(self, corpus_path, n, embed_dim=50, hidden_dim=100, context_size=3):
        self.corpus_path = corpus_path
        self.n = n
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.tokenizer = Tokenizer()
        self.vocab = None
        self.idx_to_word = None
        self.model = None

    def load_corpus(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        return self.tokenizer.split_into_sentences(text)

    def create_ngrams(self, sentences):
        ngrams = []
        for sentence in sentences:
            tokenized_sentences = self.tokenizer.tokenize(sentence)  
            for words in tokenized_sentences:
                if len(words) < self.n:
                    continue
                for i in range(len(words) - self.n + 1):
                    ngrams.append((tuple(words[i:i+self.n-1]), words[i+self.n-1]))
        return ngrams

    def build_vocab(self, ngrams):
        word_counts = Counter()
        for context, target in ngrams:
            word_counts.update(context)
            word_counts[target] += 1
        word_counts["<unk>"] = 10  
        vocab = {word: i for i, (word, _) in enumerate(word_counts.most_common())}
        idx_to_word = {i: word for word, i in vocab.items()}
        return vocab, idx_to_word

    def encode_data(self, ngrams):
        X, y = [], []
        for context, target in ngrams:
            X.append([self.vocab.get(word, self.vocab["<unk>"]) for word in context])
            y.append(self.vocab.get(target, self.vocab["<unk>"]))
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def train(self, epochs=5, lr=0.01, model_save_path="ffnn_model.pt"):
        sentences = self.load_corpus()
        train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
        ngrams = self.create_ngrams(train_sentences)
        self.vocab, self.idx_to_word = self.build_vocab(ngrams)
        X_train, y_train = self.encode_data(ngrams)
        self.model = FFNNLanguageModel(len(self.vocab), self.embed_dim, self.hidden_dim, self.n-1)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

def main():
    parser = argparse.ArgumentParser(description="Language Model Trainer and Evaluator")
    parser.add_argument("corpus_path", type=str, help="Path to the corpus file")
    parser.add_argument("n", type=int, help="n-gram size")
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    lm = LanguageModel(args.corpus_path, args.n)

    if args.train:
        print("Training the model...")
        lm.train()

if __name__ == "__main__":
    main()
