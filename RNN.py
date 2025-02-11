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
import math
from tokenizer import Tokenizer

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)  
        output, hidden = self.rnn(embedded)  
        output = self.fc(output[:, -1, :])  
        return output  


class LanguageModel:
    def __init__(self, corpus_path, n, embed_dim=128, hidden_dim=256):
        self.corpus_path = corpus_path
        self.n = n
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.tokenizer = Tokenizer()
        self.vocab = None
        self.idx_to_word = None
        self.model = None
        self.type = 'r'  

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
        word_counts["<s>"] = 10
        word_counts["</s>"] = 10
        vocab = {word: i for i, (word, _) in enumerate(word_counts.most_common())}
        idx_to_word = {i: word for word, i in vocab.items()}
        return vocab, idx_to_word

    def encode_data(self, ngrams):
        X, y = [], []
        for context, target in ngrams:
            X.append([self.vocab.get(word, self.vocab["<unk>"]) for word in context])
            y.append(self.vocab.get(target, self.vocab["<unk>"]))
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def train(self, epochs=5, lr=0.001, model_save_path="rnn_model.pt"):
        corpus_path_variable = ""
        model_variable = ""

        if self.corpus_path == "pride_and_prejudice.txt":
            corpus_path_variable = "pride"
        elif self.corpus_path == "ulysses.txt":
            corpus_path_variable = "ulysses"
        else:
            print("Invalid corpus path. Please provide a valid corpus path.")
            return

        if self.type == "r":
            model_variable = "rnn"
        else:
            print("Invalid model type. Please provide a valid model type.")
            return

        model_save_path = f"{model_variable}_model_{self.n}_{corpus_path_variable}.pt"

        if os.path.exists(model_save_path):
            print(f"Model already exists at {model_save_path}. Loading model...")
            self.load_model(model_save_path)
            return

        print("Training the model...")
        sentences = self.load_corpus()
        train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)
        self.train_sentences = train_sentences
        self.test_sentences = test_sentences
        ngrams = self.create_ngrams(train_sentences)
        self.vocab, self.idx_to_word = self.build_vocab(ngrams)
        X_train, y_train = self.encode_data(ngrams)
        self.model = RNNLanguageModel(len(self.vocab), self.embed_dim, self.hidden_dim)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
        self.save_model(model_save_path)
        print(f"Model saved to {model_save_path}")

    def save_model(self, model_save_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'idx_to_word': self.idx_to_word,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'n': self.n,
            'type': self.type,
            'train_sentences': self.train_sentences,
            'test_sentences': self.test_sentences,
            'corpus_path': self.corpus_path
        }, model_save_path)

    def load_model(self, model_save_path):
        checkpoint = torch.load(model_save_path)
        self.vocab = checkpoint['vocab']
        self.idx_to_word = checkpoint['idx_to_word']
        self.embed_dim = checkpoint['embed_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.n = checkpoint['n']
        self.type = checkpoint['type']
        self.train_sentences = checkpoint['train_sentences']
        self.test_sentences = checkpoint['test_sentences']
        self.corpus_path = checkpoint['corpus_path']
        self.model = RNNLanguageModel(len(self.vocab), self.embed_dim, self.hidden_dim)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def compute_perplexity(self, sentence):
        tokenized_sentence = self.tokenizer.tokenize(sentence)[0]
        if len(tokenized_sentence) < self.n:
            return None
        ngrams = []
        for i in range(len(tokenized_sentence) - self.n + 1):
            ngrams.append((tuple(tokenized_sentence[i:i+self.n-1]), tokenized_sentence[i+self.n-1]))
        X, y = self.encode_data(ngrams)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
            perplexity = torch.exp(nn.functional.cross_entropy(logits, y)).item()
        return perplexity

    def save_perplexities_to_file(self, sentences, file_path):
        perplexities = []
        for sentence in tqdm(sentences):
            perplexity = self.compute_perplexity(sentence)
            if perplexity is not None:
                perplexities.append(perplexity)
        avg_perplexity = sum(perplexities) / len(perplexities)
        with open(file_path, 'w') as f:
            f.write(f"{avg_perplexity}\n")
            for sentence, perplexity in zip(sentences, perplexities):
                f.write(f"{sentence}\t{perplexity}\n")

    def write_perplexity_to_file(self):
        roll_number = "2021101113"
        corpus_path_variable = ""
        model_variable = ""

        if self.corpus_path == "pride_and_prejudice.txt":
            corpus_path_variable = "pride"
        elif self.corpus_path == "ulysses.txt":
            corpus_path_variable = "ulysses"
        else:
            print("Invalid corpus path. Please provide a valid corpus path.")
            return

        if self.type == "r":
            model_variable = "rnn"
        else:
            print("Invalid model type. Please provide a valid model type.")
            return

        train_path = f"{roll_number}_train_{model_variable}_{self.n}_{corpus_path_variable}.txt"
        test_path = f"{roll_number}_test_{model_variable}_{self.n}_{corpus_path_variable}.txt"
        self.save_perplexities_to_file(self.train_sentences, train_path)
        self.save_perplexities_to_file(self.test_sentences, test_path)

    def scrap_unnecessary_corpus(self, corpus: str):
        corpus = re.sub(r'CHAPTER *[A-Z]+\.', ' ', corpus)
        corpus = re.sub(r'END OF VOL\. *[A-Z]+\.', ' ', corpus)
        corpus = re.sub(r'VOL\. *[A-Z]+\.', ' ', corpus)
        corpus = re.sub(r'Section *\d+\.', ' ', corpus)
        corpus = re.sub(r'Mr\.', 'Mr', corpus)
        corpus = re.sub(r'Mrs\.', 'Mrs', corpus)
        corpus = re.sub(r'e\.g\.', 'eg', corpus, flags=re.IGNORECASE)
        corpus = re.sub(r'_?\(([^\)]*)\)_?', lambda match: f"{match.group(1)}", corpus)
        corpus = re.sub(r'—(\w*)', lambda match: f"{match.group(1)}", corpus)
        corpus = corpus.lower()
        return corpus

    def predict_next_word(self, sentence, k=5):
        sentence = self.scrap_unnecessary_corpus(sentence)
        tokenized_sentence = self.tokenizer.tokenize(sentence)[0]
        if len(tokenized_sentence) < self.n - 1:
            return "Input sentence is too short for the n-gram model."

        context = tokenized_sentence[-(self.n - 1):]
        context_indices = [self.vocab.get(word, self.vocab["<unk>"]) for word in context]
        X = torch.tensor([context_indices], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
            top_k_probs, top_k_indices = torch.topk(probs, k, dim=1)
        top_k_words = [self.idx_to_word[idx.item()] for idx in top_k_indices[0]]
        top_k_probs = top_k_probs[0].tolist()

        return list(zip(top_k_words, top_k_probs))
