# Skip-Gram with Negative Sampling (SGNS)

This repository contains a custom implementation of the Skip-Gram model with Negative Sampling (SGNS), used to learn word embeddings from a corpus. The model is implemented from scratch using NumPy, without relying on external machine learning libraries.

## 📁 Repository Structure

- `code.py` - Main implementation of the SGNS model.
- `test.py` - Unit tests and example usages of the model.
- `drSeuss.txt` - Sample corpus from Dr. Seuss books.
- `harrypotter1.txt` - Sample corpus from the first Harry Potter book.

## 🚀 Features

- Learns word embeddings using Skip-Gram with Negative Sampling.
- Supports:
  - Cosine similarity between words.
  - Finding closest words to a given word.
  - Analogy resolution (`king` - `man` + `woman` ≈ `queen`).
- Allows combining target and context embeddings in different ways.
- Early stopping based on loss monitoring.
- Save/load models using pickle.

## 🛠️ Requirements

- Python 3.6+
- nltk
- numpy
- pandas

Additionally, certain NLTK components such as tokenizers, stopword lists, and lemmatizers must be downloaded manually.

---

## ✂️ Text Preprocessing

Before training, input text is preprocessed to normalize and clean the data. This includes:

- Converting all text to lowercase
- Removing punctuation and non-alphabetic characters
- Tokenizing the text into sentences and words
- Removing common stopwords (e.g., “the”, “is”, “and”)
- Lemmatizing words to reduce them to their base form

The result is a list of cleaned, tokenized sentences suitable for training.

---

## 🧠 Model Training

The SGNS model builds a vocabulary from the text and generates word-context pairs using a specified window size. For each word, it creates positive training samples (word and its context words) and negative samples (random words from the vocabulary). 

Embeddings are updated via stochastic gradient descent over several epochs. The model stores separate embedding matrices for target and context words.

---

## 🔍 Embedding Queries

Once trained, the model supports several types of queries:

- **Similarity computation**: Calculates the cosine similarity between two word vectors.
- **Nearest neighbor search**: Finds the most similar words to a given word based on vector distance.
- **Analogy solving**: Uses vector arithmetic to find analogies (e.g., “man is to king as woman is to ___”).

The model also allows combining target and context embeddings using options such as averaging or concatenation.

---

## 💾 Saving and Loading

Trained models can be saved to disk and reloaded later for querying without retraining. This enables efficient reuse across sessions.

---

## ✅ Testing

The `test.py` script validates model functionality by performing the following:

- Training the model on one of the included corpora
- Printing word similarity results
- Listing nearest neighbors
- Evaluating analogy queries

---

## ⚙️ Model Parameters

You can configure the following parameters when initializing or training the model:

| Parameter     | Description                                 |
|---------------|---------------------------------------------|
| `d`           | Dimension of the word embeddings            |
| `context`     | Size of the context window                  |
| `neg_samples` | Number of negative samples per target word  |
| `min_count`   | Minimum frequency for a word to be included in the vocabulary |
| `epochs`      | Number of passes over the training data     |
| `step_size`   | Learning rate for gradient updates          |
| `combo`       | Method for combining target and context vectors (target only, average, or concatenation) |

---

## 📚 Training Corpora

Two sample text files are included for experimentation:

- `drSeuss.txt` - Light, rhythmic children’s language.
- `harrypotter1.txt` - Narrative-style prose with richer vocabulary.

These can be replaced with any plain `.txt` file for training the model on different domains or writing styles.

---

## 🚀 Future Improvements

Possible enhancements to the project include:

- Adding subsampling to reduce the impact of frequent words
- Supporting Continuous Bag-of-Words (CBOW) architecture
- Visualizing word embeddings in 2D (e.g., using t-SNE)
- Comparing performance against pretrained vectors such as GloVe or word2vec

---

## 👨‍💻 Author

Developed as a learning project for understanding word embeddings and training language models from scratch using NumPy. Contributions and suggestions are welcome.

---
