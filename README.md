## generator.py

**Usage:**
```sh
generator.py <lm_type> <corpus_path> <k>
```

- **lm_type**: Specifies the type of language model. The options are:
  - `f`: Feedforward Neural Network (FFNN)
  - `r`: Recurrent Neural Network (RNN)
  - `l`: Long Short-Term Memory (LSTM)
- **corpus_path**: Denotes the file path of the respective dataset.
- **k**: Denotes the number of candidates for the next word to be printed.

### Output
On running the file, the script provides a prompt that:
- Asks for a sentence as input.
- Outputs the most probable next word of the sentence along with its probability score using the specified language model.

### Example
```sh
python3 generator.py -f ./corpus.txt 3
```
**Input sentence:**
```
An apple a day keeps the doctor
```
**Output:**
```
away 0.4
happy 0.2
fresh 0.1
```