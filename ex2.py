import pickle
import numpy as np
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import os, time, re, sys, random, math, collections, nltk

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Nadav Toledo', 'id': '209496009', 'email': 'nadavtol@post.bgu.ac.il'}


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.


    Args:
        fn: full path to the text file to process
    """
    try:
        with open(fn, 'r', encoding='utf-8') as file:
            corpus = file.read().lower()
    except UnicodeDecodeError:
        with open(fn, 'r', encoding='windows-1252') as file:
            corpus = file.read().lower()

    # Clean punctuation but keep sentence boundaries
    corpus = re.sub(r'[^a-zA-Z0-9.?!\s]', '', corpus)

    # Sentence tokenization
    raw_sentences = sent_tokenize(corpus)

    # Initialize tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_sentences = []

    for sentence in raw_sentences:
        words = word_tokenize(sentence)
        words = [
            lemmatizer.lemmatize(word)
            for word in words
            if word.isalnum() and word not in stop_words
        ]
        if words:
            cleaned_sentences.append(' '.join(words))

    return cleaned_sentences


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.

    Args:
        fn: the full path to the model to load.
    """

    file = open(fn, "rb")
    sg_model = pickle.load(file)
    file.close()
    return sg_model


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context  # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold  # ignore low frequency words (appearing under the threshold)

        self.word_counts = collections.Counter(' '.join(sentences).lower().split())
        self.word_counts = {w: c for w, c in self.word_counts.items() if c >= word_count_threshold}
        self.word_to_index = {w: i for i, w in enumerate(self.word_counts)}
        self.index_to_word = {i: w for w, i in self.word_to_index.items()}
        self.vocab_size = len(self.word_to_index)

        self.T = None
        self.C = None
        self.V = None

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.

        Args:
            w1: a word
            w2: a word
        Returns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
    """

        try:
            idx1 = self.word_to_index[w1.lower()]
            idx2 = self.word_to_index[w2.lower()]
            vec1 = self.V[:, idx1]
            vec2 = self.V[:, idx2]
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return sim
        except KeyError:
            return 0.0

    def get_closest_words(self, w, n=5):
        """
        Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """

        w = w.lower()
        if w not in self.word_to_index:
            return []
        idx = self.word_to_index[w]
        vec = self.V[:, idx]
        sims = {}
        for other_word, i in self.word_to_index.items():
            if other_word == w:
                continue
            other_vec = self.V[:, i]
            sim = np.dot(vec, other_vec) / (np.linalg.norm(vec) * np.linalg.norm(other_vec))
            sims[other_word] = sim

        return sorted(sims, key=sims.get, reverse=True)[:n]

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """
        Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """

        T = np.random.randn(self.d, self.vocab_size).astype(np.float64)
        C = np.random.randn(self.vocab_size, self.d).astype(np.float64)
        training_data = self._generate_training_data()
        neg_sampling_table = list(self.word_counts.keys())
        best_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(training_data)

            for target, context in training_data:
                if target not in self.word_to_index or context not in self.word_to_index:
                    continue

                i_target = self.word_to_index[target]
                i_context = self.word_to_index[context]

                t_vec = np.asarray(T[:, i_target], dtype=np.float64)
                c_vec = np.asarray(C[i_context, :], dtype=np.float64)

                # Positive sample
                pos_score = sigmoid(np.dot(c_vec, t_vec))
                loss = -np.log(np.clip(pos_score, 1e-10, 1.0))
                grad = pos_score - 1

                C[i_context, :] = c_vec - step_size * grad * t_vec
                T[:, i_target] = t_vec - step_size * grad * c_vec
                total_loss += loss

                # Negative samples
                for _ in range(self.neg_samples):
                    neg_word = random.choices(neg_sampling_table, weights=self.word_counts.values())[0]
                    if neg_word not in self.word_to_index:
                        continue
                    i_neg = self.word_to_index[neg_word]
                    neg_vec = np.asarray(C[i_neg, :], dtype=np.float64)

                    neg_score = sigmoid(np.dot(neg_vec, t_vec))
                    loss_neg = -np.log(np.clip(1 - neg_score, 1e-10, 1.0))
                    grad = neg_score

                    C[i_neg, :] = neg_vec - step_size * grad * t_vec
                    T[:, i_target] = T[:, i_target] - step_size * grad * neg_vec

                    total_loss += loss_neg

            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping:
                    print("Early stopping triggered.")
                    break

        self.T, self.C = T, C

        if model_path:
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)

        return T, C

    def combine_vectors(self, T, C, combo=0, model_path=None):
        """
        Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a point wise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        if combo == 0:
            V = T
        elif combo == 1:
            V = C.T
        elif combo == 2:
            V = (T + C.T) / 2
        elif combo == 3:
            V = T + C.T
        elif combo == 4:
            V = np.vstack([T, C.T])
        else:
            raise ValueError("Invalid combo value.")
        self.V = V
        if model_path:
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
        return V

    def find_analogy(self, w1, w2, w3):
        """
        Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        try:
            idx1 = self.word_to_index[w1.lower()]
            idx2 = self.word_to_index[w2.lower()]
            idx3 = self.word_to_index[w3.lower()]
        except KeyError:
            return ''  # One of the words is not in vocab

        if self.V is None:
            self.combine_vectors(self.T, self.C, combo=0)  # Default combine method

        v1 = self.V[:, idx1]
        v2 = self.V[:, idx2]
        v3 = self.V[:, idx3]

        # Compute analogy vector
        target = v2 - v1 + v3

        # Normalize target vector
        norm_target = np.linalg.norm(target)
        if norm_target == 0:
            return ''
        target = target / norm_target

        best_word = ''
        best_sim = -1

        for word, idx in self.word_to_index.items():
            if word in {w1.lower(), w2.lower(), w3.lower()}:
                continue
            vec = self.V[:, idx]
            norm_vec = np.linalg.norm(vec)
            if norm_vec == 0:
                continue
            sim = np.dot(vec, target) / norm_vec

            if sim > best_sim:
                best_sim = sim
                best_word = word

        return best_word

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """
        Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """

        try:
            idx1 = self.word_to_index[w1.lower()]
            idx2 = self.word_to_index[w2.lower()]
            idx3 = self.word_to_index[w3.lower()]
        except KeyError:
            return False

        target = self.V[:, idx2] - self.V[:, idx1] + self.V[:, idx3]
        similarities = {
            word: np.dot(self.V[:, idx], target) / (np.linalg.norm(self.V[:, idx]) * np.linalg.norm(target))
            for word, idx in self.word_to_index.items() if word not in {w1, w2, w3}
        }
        top_n = sorted(similarities, key=similarities.get, reverse=True)[:n]
        return w4.lower() in top_n

    def _generate_training_data(self):
        data = []
        for sentence in self.sentences:
            tokens = [w for w in sentence.lower().split() if w in self.word_to_index]
            for i, target in enumerate(tokens):
                for j in range(max(0, i - self.context), min(len(tokens), i + self.context + 1)):
                    if i != j:
                        context = tokens[j]
                        data.append((target, context))
        return data
