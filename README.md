# Text Classification using Neural Network

The [NTC-SCV](https://streetcodevn.com) dataset consists of 50,000 samples, with 30,000 samples for training, 10,000 samples for validation, and 10,000 samples for testing. There are two labels: positive and negative.

## Download dataset
```python
!git clone https://github.com/congnghia0609/ntc-scv.git
!unzip ./ntc-scv/data/data_test.zip -d ./data
!unzip ./ntc-scv/data/data_train.zip -d ./data
!rm -rf ./ntc-scv
```
```python
import os
import pandas as pd

def load_data_from_path(folder_path):
    examples = []
    for label in os.listdir(folder_path):
        full_path = os.path.join(folder_path, label)
        for file_name in os.listdir(full_path):
            file_path = os.path.join(full_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                sentence = " ".join(lines)
                if label == "neg":
                    label = 0
                elif label == "pos":
                    label = 1
                data = {
                    'sentence': sentence,
                    'label': label
                }
                examples.append(data)
    return pd.DataFrame(examples)

folder_paths = {
    'train': './data/data_train/train',
    'valid': './data/data_train/test',
    'test': './data/data_test/test'
}

train_df = load_data_from_path(folder_paths['train'])
valid_df = load_data_from_path(folder_paths['valid'])
test_df = load_data_from_path(folder_paths['test'])
```
## Data preprocessing:
For the NTC-SCV dataset, data preprocessing involves two steps:
- Remove non-Vietnamese comments.
- Clean the data:
-- Removing HTML tags and URLs.
-- Removing punctuation and numbers.
-- Removing special characters, emoticons, etc.
-- Normalizing whitespace.
-- Converting text to lowercase.
```python
# Install library
!pip install langid

from langid.langid import LanguageIdentifier, model

def identify_vn(df):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    not_vi_idx = set()
    THRESHOLD = 0.9

    for idx, row in df.iterrows():
        sentence = row["sentence"]
        lang, _ = identifier.classify(sentence)

        if lang != "vi" or (lang == "vi" and _ <= THRESHOLD):
            not_vi_idx.add(idx)

    vi_df = df[~df.index.isin(not_vi_idx)]
    not_vi_df = df[df.index.isin(not_vi_idx)]

    return vi_df, not_vi_df

train_df_vi, train_df_other = identify_vn(train_df)
```
```python
import re
import string

def preprocess_text(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub(' ', text)
    html_pattern = re.compile(r'<.*?>')
    text = html_pattern.sub(' ', text)
    replace_chars = string.punctuation + string.digits
    for char in replace_chars:
        text = text.replace(char, ' ')
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U0001F1F2-\U0001F1F4"  # Macau flag
                            u"\U0001F1E6-\U0001F1FF"  # flags
                            u"\U0001F600-\U0001F64F"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U0001F1F2"
                            u"\U0001F1F4"
                            u"\U0001F620"
                            u"\u200d"
                            u"\u2640-\u2642"
                            "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(' ', text)
    text = ' '.join(text.split())
    text = text.lower()
    return text

train_df_vi['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in train_df_vi.iterrows()]
valid_df['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in valid_df.iterrows()]
test_df['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in test_df.iterrows()]
```
## Converting text data into vectors
To represent text data as features (vectors), we use the torchtext library.
```python
# Install a specific version of TorchText
!pip install -q torchtext==0.16.0

# Import necessary libraries
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

# Create a tokenizer based on the English language
tokenizer = get_tokenizer("basic_english")

# Vocabulary creation function
def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)

# Vocabulary size
vocab_size = 10000

# Build the vocabulary from the data
vocabulary = build_vocab_from_iterator(
    yield_tokens(train_df_vi['preprocess_sentence'], tokenizer),
    max_tokens=vocab_size,
    specials=["<unk>"]
)

# Set the default index for the vocabulary
vocabulary.set_default_index(vocabulary["<unk>"])

# Convert the data into TorchText dataset format
def prepare_dataset(df):
    for index, row in df.iterrows():
        sentence = row['preprocess_sentence']
        encoded_sentence = [vocabulary[token] for token in tokenizer(sentence)]
        label = row['label']
        yield encoded_sentence, label

# Prepare the training data
train_dataset = list(prepare_dataset(train_df_vi))
train_dataset = to_map_style_dataset(train_dataset)

# Prepare the validation data
valid_dataset = list(prepare_dataset(valid_df))
valid_dataset = to_map_style_dataset(valid_dataset)
```
## Building a data pipeline with Dataset and DataLoader
```python
import torch
from torch.utils.data import DataLoader

# Define the device (CPU or CUDA)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a collate function to process batches
def collate_batch(batch):
    encoded_sentences, labels, offsets = [], [], [0]

    for encoded_sentence, label in batch:
        labels.append(label)
        encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.int64)
        encoded_sentences.append(encoded_sentence)
        offsets.append(encoded_sentence.size(0))

    labels = torch.tensor(labels, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    encoded_sentences = torch.cat(encoded_sentences)

    return encoded_sentences.to(device), offsets.to(device), labels.to(device)

# Batch size
batch_size = 128

# Create DataLoader for training dataset
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)

# Create DataLoader for validation dataset
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)
```
## Build a classification model
Construct the model with the following layers:
- EmbeddingBag: This layer converts words with corresponding indices into vector representations with dimension embedding_dim. It then computes the average value of the vectors in the sentence to create a representative vector for each text.
- Linear: This layer transforms the representative vectors for each text into an output with 2 nodes for the binary classification problem.
```python
from torch import nn

class TextClassificationModel(nn.Module):
  def __init__(self, vocab_size, embed_dim, num_class):
    super(TextClassificationModel, self).__init__()
    self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
    self.fc = nn.Linear(embed_dim, num_class)
    self.init_weights()

  def init_weights(self):
    initrange = 0.5
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.fc.weight.data.uniform_(-initrange, initrange)
    self.fc.bias.data.zero_()

  def forward(self, inputs, offsets):
    embedded = self.embedding(inputs, offsets)
    return self.fc(embedded)

num_class = len(set(train_df_vi['label']))
vocab_size = len(vocabulary)
embed_dim = 100
model = TextClassificationModel(vocab_size, embed_dim, num_class).to(device)
```
## Train the model
- Define the model training function:
```python
import time

def train(model, optimizer, criterion, train_dataloader, epoch=0, log_interval=25):
  model.train()
  total_acc, total_count = 0, 0
  losses = []
  start_time = time.time()
  for idx, (inputs, offsets, labels) in enumerate(train_dataloader):
    optimizer.zero_grad()
    predictions = model(inputs, offsets)

    # compute loss
    loss = criterion(predictions, labels)
    losses.append(loss.item())

    # backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()

    total_acc += (predictions.argmax(1) == labels).sum().item()
    total_count += labels.size(0)

    if idx % log_interval == 0 and idx > 0:
        elapsed = time.time() - start_time
        print(
            "| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}".format(
                epoch, idx, len(train_dataloader), total_acc/total_count
                )
            )
        total_acc, total_count = 0, 0
        start_time = time.time()

  epoch_acc = total_acc / total_count
  epoch_loss = sum(losses) / len(losses)
  return epoch_acc, epoch_loss
```
- Define the evaluation function:
```python
def evaluate(model, criterion, valid_dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, offsets, labels) in enumerate(valid_dataloader):
            predictions = model(inputs, offsets)
            loss = criterion(predictions, labels)
            losses.append(loss)

            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    epoch_acc = total_acc / total_count
    epoch_loss = sum(losses) / len(losses)

    return epoch_acc, epoch_loss
```
- Train the model:
```python
learning_rate = 5
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 5
for epoch in range(1, num_epochs+1):
  epoch_start_time = time.time()
  train_acc, train_loss = train(model, optimizer, criterion, train_dataloader)

  eval_acc, eval_loss = evaluate(model, criterion, valid_dataloader)

  print("-"*59)
  print(
      "| End of epoch {:3d} | Time: {:5.2f}s | Train Accuracy {:8.3f} | Train Loss {:8.3f} "
      "| Valid Accuracy {:8.3f} | Valid Loss {:8.3f} ".format(
          epoch, time.time() - epoch_start_time, train_acc, train_loss, eval_acc, eval_loss
      )
  )
  print("-"*59)
```
```
| epoch   0 |    25/  233 batches | accuracy    0.592
| epoch   0 |    50/  233 batches | accuracy    0.757
| epoch   0 |    75/  233 batches | accuracy    0.797
| epoch   0 |   100/  233 batches | accuracy    0.818
| epoch   0 |   125/  233 batches | accuracy    0.816
| epoch   0 |   150/  233 batches | accuracy    0.840
| epoch   0 |   175/  233 batches | accuracy    0.838
| epoch   0 |   200/  233 batches | accuracy    0.848
| epoch   0 |   225/  233 batches | accuracy    0.842
-----------------------------------------------------------
| End of epoch   1 | Time:  1.37s | Train Accuracy    0.845 | Train Loss    0.460 | Valid Accuracy    0.835 | Valid Loss    0.393 
-----------------------------------------------------------
| epoch   0 |    25/  233 batches | accuracy    0.867
| epoch   0 |    50/  233 batches | accuracy    0.851
| epoch   0 |    75/  233 batches | accuracy    0.862
| epoch   0 |   100/  233 batches | accuracy    0.858
| epoch   0 |   125/  233 batches | accuracy    0.869
| epoch   0 |   150/  233 batches | accuracy    0.855
| epoch   0 |   175/  233 batches | accuracy    0.872
| epoch   0 |   200/  233 batches | accuracy    0.858
| epoch   0 |   225/  233 batches | accuracy    0.855
-----------------------------------------------------------
| End of epoch   2 | Time:  1.17s | Train Accuracy    0.874 | Train Loss    0.359 | Valid Accuracy    0.864 | Valid Loss    0.354 
-----------------------------------------------------------
| epoch   0 |    25/  233 batches | accuracy    0.867
| epoch   0 |    50/  233 batches | accuracy    0.867
| epoch   0 |    75/  233 batches | accuracy    0.872
| epoch   0 |   100/  233 batches | accuracy    0.871
| epoch   0 |   125/  233 batches | accuracy    0.874
| epoch   0 |   150/  233 batches | accuracy    0.873
| epoch   0 |   175/  233 batches | accuracy    0.873
| epoch   0 |   200/  233 batches | accuracy    0.873
| epoch   0 |   225/  233 batches | accuracy    0.864
-----------------------------------------------------------
| End of epoch   3 | Time:  1.42s | Train Accuracy    0.877 | Train Loss    0.337 | Valid Accuracy    0.870 | Valid Loss    0.342 
-----------------------------------------------------------
| epoch   0 |    25/  233 batches | accuracy    0.885
| epoch   0 |    50/  233 batches | accuracy    0.881
| epoch   0 |    75/  233 batches | accuracy    0.881
| epoch   0 |   100/  233 batches | accuracy    0.876
| epoch   0 |   125/  233 batches | accuracy    0.877
| epoch   0 |   150/  233 batches | accuracy    0.881
| epoch   0 |   175/  233 batches | accuracy    0.865
| epoch   0 |   200/  233 batches | accuracy    0.880
| epoch   0 |   225/  233 batches | accuracy    0.885
-----------------------------------------------------------
| End of epoch   4 | Time:  2.36s | Train Accuracy    0.851 | Train Loss    0.322 | Valid Accuracy    0.870 | Valid Loss    0.354 
-----------------------------------------------------------
| epoch   0 |    25/  233 batches | accuracy    0.880
| epoch   0 |    50/  233 batches | accuracy    0.879
| epoch   0 |    75/  233 batches | accuracy    0.890
| epoch   0 |   100/  233 batches | accuracy    0.876
| epoch   0 |   125/  233 batches | accuracy    0.883
| epoch   0 |   150/  233 batches | accuracy    0.890
| epoch   0 |   175/  233 batches | accuracy    0.874
| epoch   0 |   200/  233 batches | accuracy    0.888
| epoch   0 |   225/  233 batches | accuracy    0.874
-----------------------------------------------------------
| End of epoch   5 | Time:  2.21s | Train Accuracy    0.882 | Train Loss    0.314 | Valid Accuracy    0.870 | Valid Loss    0.339 
-----------------------------------------------------------
```
## Predict and evaluate the model
- Define the prediction function:
```python
def predict(text):
    with torch.no_grad():
        encoded = torch.tensor(vocabulary(tokenizer(text)))
        output = model(encoded, torch.tensor([0]))
        return output.argmax(1).item()
```
- Evaluate accuracy on the test set:
```python
predictions, labels = [], []

for index, row in test_df.iterrows():
    sentence = row['preprocess_sentence']
    label = row['label']
    prediction = predict(sentence)
    predictions.append(prediction)
    labels.append(label)

accuracy = sum(torch.tensor(predictions) == torch.tensor(labels))/len(labels)
print(f"Accuracy: {accuracy}")
```
## References

```
Accuracy: 0.8765000104904175
```

