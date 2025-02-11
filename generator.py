import argparse
from FFNN import LanguageModel as FFNNLanguageModel
from RNN import LanguageModel as RNNLanguageModel
from LSTM import LanguageModel as LSTMLanguageModel

def main():
    parser = argparse.ArgumentParser(description="Language Model Generator")
    parser.add_argument("lm_type", type=str, choices=["f", "r", "l"], help="Type of language model (-f: FFNN, -r: RNN, -l: LSTM)")
    parser.add_argument("corpus_path", type=str, help="Path to the corpus file")
    parser.add_argument("k", type=int, help="Number of candidates for the next word")
    parser.add_argument("n", type=int, help="n-gram size")
    args = parser.parse_args()


    if args.lm_type == "f":
        lm = FFNNLanguageModel(args.corpus_path, args.n)
    elif args.lm_type == "r":
        lm = RNNLanguageModel(args.corpus_path, args.n)
    elif args.lm_type == "l":
        lm = LSTMLanguageModel(args.corpus_path, args.n)

    lm.train()
    while True:
        sentence = input("Enter a sentence (or 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        candidates = lm.predict_next_word(sentence, args.k)
        print("Top candidates for the next word:")
        for word, prob in candidates:
            print(f"{word}: {prob:.4f}")

if __name__ == "__main__":
    main()