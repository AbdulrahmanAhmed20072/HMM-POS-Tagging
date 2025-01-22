# POS Tagging with Hidden Markov Model (HMM)

This repository contains an implementation of Part-of-Speech (POS) tagging using Hidden Markov Models (HMM) and the Viterbi algorithm. The project demonstrates the process of tagging words in a sentence with their respective POS using statistical methods.

## Features

- **Data Preprocessing**: Clean and prepare the input corpus for training and testing.
- **Transition and Emission Probabilities**: Compute probabilities required for HMM.
- **Viterbi Algorithm**: Implement forward and backward steps to find the best sequence of POS tags.
- **Accuracy Evaluation**: Evaluate the tagging accuracy against labeled test data.

## Files

1. **`PartOfSpeechTagging.py`**: Python script containing the project implementation.
2. **Training Data**: `WSJ_02-21.pos` - Corpus for training.
3. **Test Data**: `WSJ_24.pos` - Corpus for testing.
4. **Vocabulary**: `hmm_vocab.txt` - List of valid words for tagging.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AbdulrahmanAhmed20072/HMM-POS-Tagging.git
   ```
2. Install the required Python packages:
   ```bash
   pip install numpy pandas
   ```

## Usage

1. Load the training data and vocabulary.
2. Preprocess the data using the `preprocess` function.
3. Compute transition and emission matrices.
4. Use the Viterbi algorithm for tagging sentences.
5. Evaluate the model's accuracy.

Run the script as follows:
```bash
python PartOfSpeechTagging.py
```

## Outputs

- Transition Matrix
- Emission Matrix
- POS Tags for test sentences
- Accuracy of the model

## Example

Input Sentence:
```
Time flies like an arrow.
```

Predicted Tags:
```
NN VBZ IN DT NN .
```

## Accuracy
The Viterbi algorithm achieved an accuracy of **~XX%** on the provided test data.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.
