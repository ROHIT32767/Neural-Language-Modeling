from typing import List

class N_Gram_Model:
    """
    Represents an N-gram language model.

    Attributes:
        n (int): The order of the N-gram model.
        ngram_counts (dict): A dictionary storing the counts of each N-gram. 
    """
    def __init__(self, n):
        """
        Initializes an N_Gram_Model instance.

        Args:
            n (int): The order of the N-gram model.
        """
        self.n = n
        self.ngram_counts = {}

    def train(self, corpus: List[List[str]]):
        """
        Trains the N-gram model on the given corpus.

        Args:
            corpus (List[List[str]]): A list of sentences, where each sentence is a list of tokens.
        """
        for sentence in corpus:
            for i in range(len(sentence) - self.n + 1):
                n_gram = tuple(sentence[i:i + self.n])
                if n_gram in self.ngram_counts:
                    self.ngram_counts[n_gram] += 1
                else:
                    self.ngram_counts[n_gram] = 1

    def get_all_n_grams(self):
        """
        Returns a dictionary of all n-grams and their counts.

        Returns:
            dict: A dictionary where keys are n-grams (tuples) and values are their counts.
        """
        return self.ngram_counts
    
    def __getitem__(self, n_gram):
        """
        Returns the count of the given n-gram.

        Args:
            n_gram (tuple): The n-gram to look up.

        Returns:
            int: The count of the n-gram, or 0 if the n-gram is not found.
        """
        if n_gram in self.ngram_counts:
            return self.ngram_counts[n_gram]
        else:
            return 0

    def __contains__(self, n_gram):
        """
        Checks if the given n-gram exists in the model.

        Args:
            n_gram (tuple): The n-gram to check.

        Returns:
            bool: True if the n-gram exists, False otherwise.
        """
        return n_gram in self.ngram_counts

def generate_n_gram_model(n, corpus: List[List[str]]):
    """
    Creates an N_Gram_Model instance and trains it on the given corpus.

    Args:
        n (int): The order of the N-gram model.
        corpus (List[List[str]]): A list of sentences, where each sentence is a list of tokens.

    Returns:
        N_Gram_Model: The trained N-gram model.
    """
    model = N_Gram_Model(n)
    model.train(corpus)
    return model