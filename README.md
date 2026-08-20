# LSTM RNN - Next Word Prediction

This project uses a Long Short-Term Memory (LSTM) neural network to predict the next word in a sentence. The model is trained on Shakespeare's Hamlet text and is deployed using a simple Streamlit web app.

## Files

- `experiments.py` - Builds and trains the LSTM model.
- `app.py` - Loads the trained model and tokenizer and predicts the next word in a user input.
- `hamlet.txt` - Text dataset used for training.
- `next_word_lstm.h5` - Saved trained model.
- `tokenizer.pickle` - Saved tokenizer used for preprocessing.

## How it works

1. The text is cleaned and tokenized.
2. Word sequences are created and padded for model input.
3. An LSTM model learns to predict the next word.
4. The trained model is saved and used by the Streamlit app.

## Run the project

Install dependencies:

```bash
pip install tensorflow streamlit numpy scikit-learn
```

Train the model:

```bash
python experiments.py
```

Run the app:

```bash
streamlit run app.py
```

## Purpose

This project demonstrates how an LSTM network can be used for text generation and next-word prediction in natural language processing tasks.

## Note

This trained model is a local machine experiment and is not very accurate or production-ready. It was trained on a small, limited dataset and local hardware, so its predictions are intended for learning and demonstration rather than high-quality real-world performance.
