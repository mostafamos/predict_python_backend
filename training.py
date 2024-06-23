import os
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Define the output directory
output_dir = './results'

# Read the contents of the text file
file_path = 'db_connection_info.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract text and labels from each line
texts = []
labels = []
for line in lines:
    text, label = line.strip().split(',')
    texts.append(text)
    labels.append(int(label))

# Create a DataFrame with the extracted text and labels
training_data = pd.DataFrame({
    'text': texts,
    'label': labels
})
print(training_data)

# Convert the DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(training_data)

# Load the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the training data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the tokenized dataset into training and evaluation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
