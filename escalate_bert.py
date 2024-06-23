import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned BERT model and tokenizer
output_dir = './results'
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example texts for inference
new_texts = [
    "database connection issues",  # Should be classified as data
    "firewall configuration problem",  # Should be classified as infra
    "network latency affecting all servers",  # New example more likely for infra
    "data extraction errors from database",  # New example more likely for data
    "VPN access not working for remote employees",  # New example more likely for infra
    "database schema mismatch",  
    "security breach detected in the network",  
    "inconsistent data across multiple databases", 
    "Pipeline expired",
    "out of memory"
]

# predict from those sentences
inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors="pt")

# create the model
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)

# define departments based on predicted labels
departments = {
    0: 'data',
    1: 'infra',
    2: 'engineering'
}

# print predictions with corresponding departments
for text, pred in zip(new_texts, predictions):
    predicted_department = departments[pred.item()]
    print(f"Text: {text} | Predicted Department: {predicted_department}")

# save the predictions to a file
predictions_df = pd.DataFrame({
    "text": new_texts,
    "predicted_label": predictions.tolist(),
    "predicted_department": [departments[pred.item()] for pred in predictions]
})
predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
