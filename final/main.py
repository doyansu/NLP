import numpy as np
import pandas as pd
import re
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import logging
logging.set_verbosity_error()

data = pd.read_csv('nlp-getting-started/train.csv')
test_data = pd.read_csv('nlp-getting-started/test.csv')

def keyword_preprocess(text):
    """Clean keyword by removing '%20'"""
    if pd.notnull(text):
        text = text.replace("%20", " ")
    else:
        text = ''
    return text

def remove_url(text):
    url_pattern = re.compile(r'https?://t\.co/[^\s]*')
    new_text = url_pattern.sub('', text)
    return new_text

def remove_at(text):
    at_pattern = re.compile(r'@[^\s]*')
    new_text = at_pattern.sub('', text)
    return new_text

def text_preprocess(text):
    """Clean text by removing url and @someone"""
    text = remove_url(text)
    text = remove_at(text)
    return text

# remove url and @ from text
# data['text'] = data['text'].apply(text_preprocess)
# test_data['text'] = test_data['text'].apply(text_preprocess)

# remove %20 from keyword
data['keyword'] = data['keyword'].apply(keyword_preprocess)
test_data['keyword'] = test_data['keyword'].apply(keyword_preprocess)

# combine keyword and text
data['keyword_text'] = data.apply(lambda row: row['keyword'] + ' ' + row['text'], axis=1)
test_data['keyword_text'] = test_data.apply(lambda row: row['keyword'] + ' ' + row['text'], axis=1)

train_data_dict = {
    "text": data["keyword_text"].tolist(),
    "label": data["target"].tolist()
}

test_data_dict = {
    "text": test_data["keyword_text"].tolist()
}

train_dataset = Dataset.from_dict(train_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# use dynamic padding 
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

training_args = TrainingArguments(
    "test-trainer",
    report_to='none',
    num_train_epochs=2,
    save_strategy = "epoch"
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

predictions = trainer.predict(tokenized_test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

submission = pd.DataFrame({'id':test_data['id'],'target':preds})
submission.to_csv('nlp-getting-started/mySubmission.csv', index=False)