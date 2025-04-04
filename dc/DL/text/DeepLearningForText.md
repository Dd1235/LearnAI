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

(add later)

# Chapter 2: Text Classification

## What is Text Classification?

Text classification is the process of assigning labels to text based on its content. It's used to organize and structure unstructured data.

### Applications
- Sentiment analysis
- Spam detection
- News categorization

### Types
- **Binary classification** (e.g., spam vs not spam)
- **Multi-class classification** (e.g., politics, sports, tech)
- **Multi-label classification** (e.g., genres: action, fantasy)

## Word Embeddings

Traditional encodings are inefficient and don't capture semantics.

### Word Embeddings:
- Represent words as vectors
- Similar words have similar vectors
- Example: King – Queen, Man – Woman

```python
import torch
from torch import nn

words = ["The", "cat", "sat", "on", "the", "mat"]
word_to_idx = {word: i for i, word in enumerate(words)}
inputs = torch.LongTensor([word_to_idx[w] for w in words])
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=10)
output = embedding(inputs)
print(output)
```

## Embedding in a Text Pipeline

```python
from torch.utils.data import Dataset, DataLoader

def preprocess_sentences(text):
    # Tokenization and other preprocessing steps
    pass

class TextDataset(Dataset):
    def __init__(self, encoded_sentences):
        self.data = encoded_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def text_processing_pipeline(text):
    tokens = preprocess_sentences(text)
    dataset = TextDataset(tokens)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader

text = "Your sample text here."
dataloader = text_processing_pipeline(text)
embedding = nn.Embedding(num_embeddings=10, embedding_dim=50)

for batch in dataloader:
    output = embedding(batch)
    print(output)
```

## CNNs for Text Classification

CNNs can classify short texts such as tweets.

### CNN Pipeline:

1. **Convolution layer**: Detect features, filer/kernel slides over the input, filter has a size, and the stride determines number of positions the filter moves.

2. **Pooling layer**: Reduce size

3. **Fully connected layer**: Final prediction based on previous layer outputs

```python

# dataset prepared later

import torch.nn.functional as F
import torch.optim as optim

class SentimentAnalysisCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        # trasforms combined outputs to desired output target size
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, text):
        # words to embedding
        embedded = self.embedding(text).permute(0, 2, 1)
        conved = F.relu(self.conv(embedded))
        conved = conved.mean(dim=2) # average across seqeunce length
        return self.fc(conved)

vocab = ["i", "love", "this", "book", "do", "not", "like"]
# instead of one hot or tfid as they are not good for this usecase
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(word_to_idx)
embed_dim = 10
model = SentimentAnalysisCNN(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

## Training the Model

```python
data = [(["i", "love", "this", "book"], 1), (["do", "not", "like"], 0)]
for epoch in range(10):
    for sentence, label in data:
        model.zero_grad()
        sentence = torch.LongTensor([word_to_idx.get(w, 0) for w in sentence]).unsqueeze(0) 
        outputs = model(sentence)
        label = torch.LongTensor([label])
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

for sample in book_samples:
    input_tensor = torch.tensor([word_to_idx[w] for w in samples], dtype=torch.long).unsequeeze(0)
    outputs = model(input_tensor)
    _, predicted_label = torch.max(outputs.data, 1)
    sentiment = "Positive" if predicted_label.item() == 1 else "Negative"
    print(f"Book Review: {' '.join(sample)}")
    print(f"Sentiment: {sentiment}\n")
```

## RNNs and Variants

- RNNs are good for capturing sequence and context.
- CNNs spot patters in chunks of text
- They remember past words for greater meaning
- They can detect hte sarcasm in "Oh I just love getting stuck in traffic".

### Dataset and dataloader

```Python

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        return eslf.text[id]

```

### LSTM

- RNNs cannot capture subtle nuances, and a mix of positive and negative sentiments.
- LSTM architecture: input gate, forget gate, output gate

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden.squeeze(0))
        return output

# Initialize the LSTM and the output layer with parameters
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# Initialize model with required parameters
lstm_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)

# Train the model by passing the correct parameters and zeroing the gradient
for epoch in range(10): 
    optimizer.zero_grad()
    outputs = lstm_model(X_train_seq)
    loss = criterion(outputs, y_train_seq)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
```

### GRU (Gated Recurrent Unit)

- what if you want to recognize spam? 
- "Congratulations! You have one $1000000" 
- quickly recognize spammy patterns without needing full context

```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.fc(hidden.squeeze(0))
        return output


# Complete the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)       
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        out, _ = self.gru(x, h0)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# Initialize the model
gru_model = GRUModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(gru_model.parameters(), lr=0.01)

# Train the model and backpropagate the loss after initialization
for epoch in range(15): 
    optimizer.zero_grad()
    outputs = gru_model(X_train_seq)
    loss = criterion(outputs, y_train_seq)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

_, predicted = torch.max(outputs, 1)
```

## Evaluation Metrics

Precision: out of all those predicted as positive how many are actually positive

Recall: out of all the positive how many were predicted as positive

F1 score: harmonizes precision and recall, ranges from 0 to 1

```python
from torchmetrics import Accuracy, Precision, Recall, F1Score

actual = torch.tensor([0, 1, 1, 0, 1, 0])
predicted = torch.tensor([0, 0, 1, 0, 1, 1])

accuracy = Accuracy(task="binary", num_classes=2)
precision = Precision(task="binary", num_classes=2)
recall = Recall(task="binary", num_classes=2)
f1 = F1Score(task="binary", num_classes=2)

print(f"Accuracy: {accuracy(predicted, actual)}")
print(f"Precision: {precision(predicted, actual)}")
print(f"Recall: {recall(predicted, actual)}")
print(f"F1 Score: {f1(predicted, actual)}")

# Create an instance of the metrics
accuracy = Accuracy(task="multiclass", num_classes=2)
precision = Precision(task="multiclass", num_classes=2)
recall = Recall(task="multiclass", num_classes=2)
f1 = F1Score(task="multiclass", num_classes=2)

# Generate the predictions
outputs = rnn_model(X_test_seq)
_, predicted = torch.max(outputs, 1)

# Calculate the metrics
accuracy_score = accuracy(predicted, y_test_seq)
precision_score = precision(predicted, y_test_seq)
recall_score = recall(predicted, y_test_seq)
f1_score = f1(outputs, y_test_seq)
print("RNN Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_score, precision_score, recall_score, f1_score))

# Create an instance of the metrics
accuracy = Accuracy(task="multiclass", num_classes=3)
precision = Precision(task="multiclass", num_classes=3)
recall = Recall(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3)

# Calculate metrics for the LSTM model
accuracy_1 = accuracy(y_pred_lstm, y_test)
precision_1 = precision(y_pred_lstm, y_test)
recall_1 = recall(y_pred_lstm, y_test)
f1_1 = f1(y_pred_lstm, y_test)
print("LSTM Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_1, precision_1, recall_1, f1_1))

# Calculate metrics for the GRU model
accuracy_2 = accuracy(y_pred_gru, y_test)
precision_2 = precision(y_pred_gru, y_test)
recall_2 = recall(y_pred_gru, y_test)
f1_2 = f1(y_pred_gru, y_test)
print("GRU Model - Accuracy: {}, Precision: {}, Recall: {}, F1 Score: {}".format(accuracy_2, precision_2, recall_2, f1_2))
```


> Extra: Add tokenization, stemming, padding, and batching to scale the pipeline.

## Summary

to be added

# Chapter 3: Text Generation

## Introduction to Text Generation

Text generation tasks include:
- Chatbots
- Language translation
- Creative writing (poetry, story generation)

We use RNNs, LSTMs, GRUs for such tasks due to their ability to maintain temporal dependencies in sequential data.

---

## RNN for Character-Level Text Generation

```python
import torch
import torch.nn as nn

data = "Hello how are you?"
chars = list(set(data))
char_to_ix = {char: i for i, char in enumerate(chars)}
ix_to_char = {i: char for i, char in enumerate(chars)}

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        # last time steps output, through the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel(1, 16, 1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```

### Preparing Inputs

```python
inputs = [char_to_ix[ch] for ch in data[:-1]]
targets = [char_to_ix[ch] for ch in data[1:]]

inputs = torch.tensor(inputs, dtype=torch.long).view(-1, 1)
inputs = nn.functional.one_hot(inputs, num_classes=len(chars)).float()
targets = torch.tensor(targets, dtype=torch.long)
```

### Training the Model

```python
for epoch in range(100):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/100, Loss: {loss.item()}')
```

### Testing the Model

```python
model.eval()
test_input = char_to_ix['h']
test_input = nn.functional.one_hot(torch.tensor(test_input).view(-1, 1), num_classes=len(chars)).float()
predicted_output = model(test_input)
predicted_char_ix = torch.argmax(predicted_output, 1).item()
print(f"Test Input: h, Predicted Output: {ix_to_char[predicted_char_ix]}")
```

---

## GANs for Text Generation

- make synthetic data that preserves statistical similarity
- Two Components:
    - Generator: creates fake samples by adding noise
    - Discriminator: differentiates between real and generated text data


```python

# embedding reviews, convert reviews to tensors

# generator network
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length, seq_length),
            nn.Sigmoid() # suitable for binary data
        )

    def forward(self, x):
        return self.model(x)

# discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(seq_length, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
```

### Training Setup

```python
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss() # binary cross entropy

# two optimizers for each
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=0.001)
```

### Training the GAN

```python
num_epochs = 50
for epoch in range(num_epochs):
    for real_data in data:
        real_data = real_data.unsqueeze(0)
        # random noise
        noise = torch.rand((1, seq_length))
        disc_real = discriminator(real_data)
        fake_data = generator(noise)
        disc_fake = discriminator(fake_data.detach())

        loss_disc = criterion(disc_real, torch.ones_like(disc_real)) +                     criterion(disc_fake, torch.zeros_like(disc_fake))
        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        disc_fake = discriminator(fake_data)
        loss_gen = criterion(disc_fake, torch.ones_like(disc_fake))
        optimizer_gen.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}: Generator loss: {loss_gen.item()}, Discriminator loss: {loss_disc.item()}")
```

### Sample Generated Output

```python
for _ in range(5):
    noise = torch.rand((1, seq_length))
    generated_data = generator(noise)
    print(torch.round(generated_data).detach())
```

---

## Pre-trained Models: GPT-2 and T5

Pre-trained models:
- trained on extensive datasets + high performance
- high computational cost + limited customization options

- hugging face + pytroch

### GPT-2 for Text Generation

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
model = GPT2LMHeadModel.from_pretrained('gpt2')

seed_text = "Once upon a time" # story's opening Line 

# tensor in pytorch format
input_ids = tokenizer.encode(seed_text, return_tensors='pt')

# temperature: randomness of the output
# prevents consecutive word repetition
# pads if less than 40
output = model.generate(input_ids, max_length=40, temperature=0.7, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)


generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### T5 for Language Translation

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_prompt = "translate English to French: 'Hello, how are you?'"
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

# accomodate longer translation if needed
output = model.generate(input_ids, max_length=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)
```
- GPT 2: text generation
- DistilGPT-2: smaller
- BERT: text calssification question answering
- T5: language translation, summarization
---

## Evaluation Metrics for Text Generation

- standard metrics fall short

### BLEU Score (Bilingual Evaluation Understudy)

- checks for the occurence of n get_feature_names_out
- n grams: consecute n word phrases in the sentence
- more the generated n grams match reference, higher the score


```python
from torchmetrics.text import BLEUScore

generated_text = ['the cat is on the mat']
real_text = [['there is a cat on the mat', 'a cat is on the mat']]

bleu = BLEUScore()
bleu_score = bleu(generated_text, real_text)
print("BLEU Score:", bleu_score.item())
```

### ROUGE Score (Recall-oreinted Understudy for Gisting Evaluation)

- compares a generated text to a reference text
- ROUGE-N: considers overlapping n-grams in both texts
- ROUGE-L: looks at the LCS between the texts

- Route metrics:
    - F-measure: Harmonic mean of precision and reecall
    - Precison: matches of n-grams in generated text within the reference text
    - Recall: matches of n-grams in reference text within the generated text

```python
from torchmetrics.text import ROUGEScore

generated_text = "Hello, how are you doing?"
real_text = "Hello, how are you?"

rouge = ROUGEScore()
rouge_score = rouge([generated_text], [[real_text]])
print("ROUGE Score:", rouge_score)
```

### Considerations and limitations

- evaluate word presence not semantic understanding
- sensitive to length

---

## Final Notes

- RNNs are simple but effective for small text generation tasks.
- GANs are more experimental for text but show promise.
- Pre-trained models like GPT-2, T5 are state-of-the-art and easy to use via HuggingFace.
- Evaluate with BLEU and ROUGE, not accuracy.

> Tip: For long sequences and memory efficiency, prefer LSTM or GRU over vanilla RNNs.

---
## Summary

to be done

# Chapter 4: Advanced Topics in Deep Learning for Text with PyTorch

## Transfer Learning

### What is it?
Transfer learning involves reusing a pretrained model on a new but related task. Instead of training a model from scratch:
- Load a pretrained model like BERT.
- Fine-tune on your specific dataset.

### Why it's useful:
- Reduces training time
- Needs less data
- Learns from general linguistic knowledge

---

## Pretrained Model: BERT

- **BERT**: Bidirectional Encoder Representations from Transformers
- Learns context in both directions
- Useful for classification, question answering, etc.

### Example Code
```python
texts = ["I love this!", "This is terrible.", "Amazing experience!", "Not my cup of tea."]
labels = [1, 0, 1, 0]

from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
inputs["labels"] = torch.tensor(labels)
```

---

### Fine-tuning
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()

for epoch in range(1):
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward() # find gradients
    optimizer.step() # weights adjusted
    optimizer.zero_grad() # gradients reset
```

### Evaluation
```python
text = "I had an awesome day!"
input_eval = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs_eval = model(**input_eval)

pred = torch.nn.functional.softmax(outputs_eval.logits, dim=-1)
label = 'positive' if torch.argmax(pred) > 0 else 'negative'
print(f"Sentiment: {label}")
```

---

## Transformers for Text

- Learn relationships across all words (global context)
- Avoid sequential bottlenecks of RNNs

### Components
- Encoder: encodes input
- Decoder: (used in generation)
- Multi-head attention
- Positional encodings
- Feed-forward networks

---

## Building Transformer Encoder

```python
import torch.nn as nn
import torch.optim as optim

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)
```

---

## Attention Mechanism

- Focuses on important tokens in the sequence.
- Helps with resolving ambiguity.
- Self-attention relates words to each other.
- Multi-head attends in multiple perspectives.

---

## Adversarial Attacks on Text Models

- FGSM: uses gradient sign to flip predictions
- PGD: multiple FGSM steps
- C&W: minimizes detection

### Defense
- Adversarial training
- Gradient masking
- Ensemble models
- Data augmentation

---

## Tools
- [Adversarial Robustness Toolbox](https://adversarial-robustness-toolbox.readthedocs.io)

---

## Summary 

to be done

---