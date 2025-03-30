# Deep Learning for Text

Chapter 1: Introduction to Deep Learning for Text with PyTorch
Chapter 2: Text Classification with pytorch
Chapter 3: Text Generation with pytorch
Chapter 4: Advanced Topics in Deep Learning for Text with pytorch

---

# Chapter 1: Introduction

Text Processing Pipeline: Raw data -> **preprocessing** -> encoding -> dataset and dataloader

---

## Preprocessing

Pytorch, NLTK: Natural Language toolkit (transform raw text to processed text)

Preprocessing techniques:
- tokenization
    - tokens/words from text
- stop word removal
    - remove common words like "a", "the", "or", etc that do not contribute to the meaning.
    - doubt: do they not add meaning? soemtimes they do? 
        - She is a lawyer, She is THE lawyer, but I suppose the second example with emphaasis on the, "THE",  is very genz

- stemming
    - reduce word to base form, eg running, ran -> run
    - doubt: are you losing information here then? or is this just a matter of preprocessing and its later taken care of.
    this technique cannot work for languages like Telugu/Kannada where in the transformed verb carreis most of the meaning

- rare word removal
    - reomve infrequent words that do not add value.
    - doubt: hmm again you might lose information? eg. "I absolutely hate how much math is involved. I hate how tedious it is. I hate how difficult it is to understand. But man do I love physics." Here "love" is infrequent but it adds so much meaning?

So as to reduce features, cleaner, more representative datasets
more techniques exist

## Preprocessing code snippets


```Python

-----------Tokenization-----------

from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("I am reading a book now. I love to read books!")

-----------Stop Word Removal-----------

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = ["I", "am", ...] # from tokenizer(...)
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

-----------Stemming-----------

import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
filtered_tokens = [...]
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

-----------Rare Word Removal-----------
from nltk.probability import FreqDist
stemmed_tokens = [...]
freq_dist = FreqDist(stemmed_tokens)
threshold = 2
common_tokens = [token for token in stemmed_tokens if freq_dist[token] > threshold]
```
---

## Encoding

- text -> machine readable numbers

### Encoding Techniques:

- One-hot encoding: word to vectors eg cat->[1,0,0]

- Bag-of-Words (BoW): captures word freq, disregarding ordered
    - doubt: how can this be that helpful? sentences structures matters more especially in languages like english, now with something like Telugu or German this can be useful because the transformed words carry more meaning

- TF-IDF: balancees uniqueness and important
- Embedding: converts words into vectors, capturing semantic meaning(chapter 2)


## Encoding: Code snippets

```Python


-----------One-hot----------
import torch

vocab = ['cat', 'dog', 'rabbit']
vocab_size = len(vocab)
one_hot_vectors = torch.eye(vocab_size)
one_hot_dict = {word: one_hot_vectors[i] for i, word in enumerate(vocab)}
print(one_hot_dict['cat'])
# tensor([1., 0., 0.])

-----------Bag of Words(with Count Vectorizer)----------

from sklearn.feature_extraction.text import CountVectorizer

# gives you frequencies of X per document

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())

-----------TF-IDF Vectorizer----------

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())
```

---

## Full Preprocessing Text Pipeline

Dataset: container for processing and encoded data
DataLoader: batching, shuffling, multiprocessing



### Dataset + DataLoader
```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]
```

### Text Processing Pipeline Functions
```python
from nltk.probability import FreqDist
import re

def encode_sentences(sentences):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer

def extract_sentences(data):
    return re.findall(r'[A-Z][^.!?]*[.!?]', data)

def preprocess_sentences(sentences):
    processed = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokens = tokenizer(sentence)
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [stemmer.stem(t) for t in tokens]
        freq_dist = FreqDist(tokens)
        tokens = [t for t in tokens if freq_dist[t] > 2]
        processed.append(' '.join(tokens))
    return processed

def text_processing_pipeline(text):
    tokens = preprocess_sentences(text)
    encoded, vectorizer = encode_sentences(tokens)
    dataset = TextDataset(encoded)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader, vectorizer
```

### Usage

```Python
text_data = "..."
sentences = extract_sentences(text_data)
dataloaders, vectorizer = [text_preprocessing_pipeline(text) for text in sentences]
print(next(iter(dataloader))[0, :10])
```

---

## Summary