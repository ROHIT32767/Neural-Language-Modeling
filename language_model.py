from tokenizer import Tokenizer
from typing import List, Dict
from abc import ABC, abstractmethod
import numpy as np

class FeedForwardNNLanguageModel(ABC):
    def __init__(self, n, threshold=2):
        self._n = n
        self.tokenizer = Tokenizer()
        self.threshold = threshold
        self.vocabulary = set()
        self.vocabulary_frequencies = {}
    
    def preprocess(self, sentence: str) -> str:
        lowercased_sentence = sentence.lower()  
        return lowercased_sentence

    def filter_sentence(self, sentence: List[str]) -> List[str]:
        prepared_sentence = []
        for token in sentence:
            if token not in self.vocabulary:
                prepared_sentence.append('<UNK>')
            elif self.vocabulary_frequencies[token] < self.threshold:
                prepared_sentence.append('<UNK>')
            else:
                prepared_sentence.append(token)

        prepared_sentence = ['<s>']*(self._n-1) + prepared_sentence + ['</s>']
        return prepared_sentence

    @abstractmethod
    def get_probability_ngram(self, n_gram: List[str]) -> float:
        pass

    def learn(self, corpus: str):
        preprocessed_corpus = self.preprocess(corpus)  # More descriptive name
        self.input_corpus = corpus
        self.corpus_text = self.tokenizer.tokenize(preprocessed_corpus)  # Use preprocessed corpus

        self.vocabulary = self.generate_vocabulary(self.corpus_text)
        self.vocabulary.add('<s>')
        self.vocabulary.add('</s>')
        self.vocabulary.add('<UNK>')

        for sentence in self.corpus_text:
            for token in sentence:
                if token not in self.vocabulary_frequencies:
                    self.vocabulary_frequencies[token] = 1
                else:
                    self.vocabulary_frequencies[token] += 1

        for word in list(self.vocabulary):
            if word == '<s>' or word == '</s>' or word == '<UNK>':
                continue
            if self.vocabulary_frequencies[word] < self.threshold:
                self.vocabulary.remove(word)

        self.corpus_text = [self.filter_sentence(sentence) for sentence in self.corpus_text]

    def get_sentence_score(self, sentence: List[str]) -> float:
        log_perplexity = 0.0
        n_gram_count = 0
        for i in range(len(sentence) - self._n + 1):
            n_gram_count += 1
            n_gram = tuple(sentence[i:i + self._n])
            proba = self.get_probability_ngram(n_gram)
            log_perplexity += np.log(max(proba, 1e-10))
        
        perplexity = np.exp(-log_perplexity / n_gram_count)
        return perplexity

    def score(self, sentence: str) -> float:
        preprocessed_sentence = self.preprocess(sentence)  # More descriptive name
        tokenized_sentence = self.tokenizer.tokenize(preprocessed_sentence)
        flattened_sentence = []
        for sublist in tokenized_sentence:
            flattened_sentence.extend(sublist)
        prepared_sentence = self.filter_sentence(flattened_sentence)
        return self.get_sentence_score(prepared_sentence)

    def predict_next_words(self, sentence, k):
        preprocessed_sentence = self.preprocess(sentence)  
        tokenized_sentence = self.tokenizer.tokenize(preprocessed_sentence)

        flattened_sentence = []
        for sublist in tokenized_sentence:
            flattened_sentence.extend(sublist)

        prepared_sentence = self.filter_sentence(flattened_sentence)
        prepared_sentence = prepared_sentence[:-1]
        prefix = tuple(prepared_sentence[-self._n + 1:])
        word_probabilities = []

        for word in self.vocabulary:
            word_proba = self.get_probability_ngram(prefix + (word,))
            word_probabilities.append((word, word_proba))

        sorted_probabilities = sorted(word_probabilities, key=lambda x: x[1], reverse=True)
        top_k_probabilities = sorted_probabilities[:k]
        normalization_factor = sum([x[1] for x in top_k_probabilities])
        normalized_probabilities = [(x[0], x[1] / normalization_factor) for x in top_k_probabilities]
        return normalized_probabilities
    
    def generate_vocabulary(self, corpus: List[List[str]]) -> set:
        vocabulary = set()
        for sentence in corpus:
            for token in sentence:
                vocabulary.add(token)
        return vocabulary
