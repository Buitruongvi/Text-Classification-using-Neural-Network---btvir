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
- Clean the data. The data cleaning steps include:
-- Removing HTML tags and URLs.
-- Removing punctuation and numbers.
-- Removing special characters, emoticons, etc.
-- Normalizing whitespace.
-- Converting text to lowercase.


