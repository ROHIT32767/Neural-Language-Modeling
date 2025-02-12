import argparse
from FFNN import LanguageModel as FFNNLanguageModel
from RNN import LanguageModel as RNNLanguageModel
from LSTM import LanguageModel as LSTMLanguageModel

def main():
    parser = argparse.ArgumentParser(description="Evaluate Perplexity Scores")
    parser.add_argument("lm_type", type=str, choices=["f", "r", "l"], help="Type of language model (-f: FFNN, -r: RNN, -l: LSTM)")
    parser.add_argument("corpus_path", type=str, help="Path to the corpus file")
    parser.add_argument("n", type=int, help="n-gram size")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model and write perplexity scores to a file")
    args = parser.parse_args()

    if args.lm_type == "f":
        lm = FFNNLanguageModel(args.corpus_path, args.n)
    elif args.lm_type == "r":
        lm = RNNLanguageModel(args.corpus_path, args.n)
    elif args.lm_type == "l":
        lm = LSTMLanguageModel(args.corpus_path, args.n)

    lm.train()

    if args.evaluate:
        lm.write_perplexity_to_file()

if __name__ == "__main__":
    main()