import sys
from typing import List
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer

class Tokenizer:
    def __init__(self):
        url_regex_pattern = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9@:%_\\+.~#?&\/=]*)?'
        hashtag_regex_pattern = r'#\w+'
        mentions_regex_pattern = r'@\w+'
        percentage_regex_pattern = r'\d+\s*\%'
        range_regex_pattern = r'\d+\s*[-–]\s*\d+'
        email_regex_pattern = r"^\S+@\S+\.\S+$"
        self.place_holders = [
            ["<URL>", url_regex_pattern],
            ["<HASHTAG>", hashtag_regex_pattern],
            ["<MENTION>", mentions_regex_pattern],
            ["<PERCENTAGE>", percentage_regex_pattern],
            ["<RANGE>", range_regex_pattern],
            ["<MAILID>", email_regex_pattern]
        ]
        self.multi_word_tokenizer = MWETokenizer([('<','URL','>'), ('<','HASHTAG','>'), ('<','MENTION','>'), ('<','PERCENTAGE','>'), ('<','RANGE','>'), ('<','MAILID','>')],separator='')

    def tokenize(self, text: str) -> List[List[str]]:
        for [substitution, pattern] in self.place_holders:
            text = re.sub(pattern, substitution, text)
        return [self.multi_word_tokenizer.tokenize(word_tokenize(sentence)) for sentence in sent_tokenize(text)]
    
    def split_into_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

if __name__ == '__main__':
    text = input('your text: ')
    tokenizer = Tokenizer()
    print('tokenized text: ', tokenizer.tokenize(text))